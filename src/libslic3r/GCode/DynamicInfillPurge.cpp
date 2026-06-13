// Dynamic infill purge: absorb multi-material purge into model sparse infill.
// See DynamicInfillPurge.hpp for the high-level description.

#include "DynamicInfillPurge.hpp"

#include "../Print.hpp"
#include "../PrintConfig.hpp"
#include "../Layer.hpp"
#include "../Surface.hpp"
#include "../ExtrusionEntity.hpp"
#include "../ExtrusionEntityCollection.hpp"
#include "ToolOrdering.hpp"
#include "WipeTower.hpp"
#include "WipeTower2.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/log/trivial.hpp>

namespace Slic3r {

namespace {

// Type1 (multi-nozzle prime tower) path clamps each transition's flush volume up
// to this minimum unless it falls below EPSILON, so per-transition savings are
// capped at max(0, matrix - this floor). Routing below the floor would become
// orphan model mass. (Type2 uses filament_minimal_purge_on_wipe_tower instead.)
constexpr float k_min_purge_volume_type1 = 100.f;

// Maximum volume of a single sparse-infill island handed to the wipe router;
// larger islands are partitioned into chunks of about this size before routing,
// so the router can spread one region's infill across several tool changes
// instead of awarding the whole island to the first reaching transition. The
// chunk pieces are spatially adjacent, so the extra inter-chunk travel is
// negligible.
constexpr double k_dip_max_island_volume_mm3 = 25.0;

// Soluble and support filaments are unabsorbable (mirrors the early-returns in
// mark_wiping_extrusions).
inline bool transition_is_absorbable(const PrintConfig& cfg,
                                     unsigned int from_filament,
                                     unsigned int to_filament)
{
    return !cfg.filament_soluble.get_at(from_filament)
        && !cfg.filament_soluble.get_at(to_filament)
        && !cfg.filament_is_support.get_at(from_filament)
        && !cfg.filament_is_support.get_at(to_filament);
}

std::vector<std::vector<float>> reshape_wipe_volumes(const std::vector<float>& flat,
                                                     size_t number_of_extruders)
{
    std::vector<std::vector<float>> out;
    out.reserve(number_of_extruders);
    for (size_t i = 0; i < number_of_extruders; ++i) {
        out.emplace_back(flat.begin() + i * number_of_extruders,
                         flat.begin() + (i + 1) * number_of_extruders);
    }
    return out;
}

// Split one infill ExtrusionPath into pieces of about target_vol mm3 each by
// cutting its polyline at cumulative-length boundaries (interpolating a vertex
// mid-segment). New pieces inherit role / mm3_per_mm / width / height via the
// (Polyline3&&, const ExtrusionPath&) constructor, so the only geometric change
// is the break points (a few extra travels between pieces). Appends the new
// pieces to `out`; falls back to a single copy when the path cannot be split.
void split_path_by_volume(const ExtrusionPath& path, double target_vol,
                          std::vector<ExtrusionEntity*>& out)
{
    const Points3& pts = path.polyline.points;
    if (pts.size() < 2 || path.mm3_per_mm <= 0.0 || path.total_volume() <= target_vol) {
        out.push_back(new ExtrusionPath(path));
        return;
    }
    // vol = mm3_per_mm * unscale(len) = mm3_per_mm * len * SCALING_FACTOR
    //  => target scaled length = target_vol / (mm3_per_mm * SCALING_FACTOR).
    const double target_len = target_vol / (path.mm3_per_mm * SCALING_FACTOR);
    auto finish = [&](Polyline3& pl) {
        if (pl.points.size() >= 2 && pl.length() > SCALED_EPSILON)
            out.push_back(new ExtrusionPath(std::move(pl), path));
    };
    Polyline3 cur;
    cur.points.push_back(pts[0]);
    double cur_len = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
        const Vec3d a   = pts[i - 1].cast<double>();
        const Vec3d b   = pts[i].cast<double>();
        const double seg = (b - a).norm();
        double done = 0.0;                       // length of this segment already placed
        // Emit cut points whenever the running piece would reach target_len.
        // Invariant: cur_len < target_len at the top of each segment, so need > 0.
        while (seg - done > 0.0 && cur_len + (seg - done) >= target_len) {
            const double need   = target_len - cur_len;
            const double cut_at = done + need;
            const double t      = cut_at / seg;
            const Vec3d  cv     = a + (b - a) * t;
            const Point3 cp(cv.x(), cv.y(), cv.z());
            cur.points.push_back(cp);
            finish(cur);
            cur = Polyline3();
            cur.points.push_back(cp);
            cur_len = 0.0;
            done = cut_at;
        }
        cur.points.push_back(pts[i]);
        cur_len += (seg - done);
    }
    finish(cur);
}

// Partition one infill island (a top-level layerm->fills.entities collection)
// into chunks whose total_volume is about target each. First explodes any
// oversized single ExtrusionPath into sub-paths (mid-polyline cuts), then groups
// the resulting pieces at whole-piece boundaries. Piece pointers are transferred
// (kept sub-entities) or freshly allocated (split sub-paths) into the returned
// collections; on return the input island is left empty and the caller must
// delete it. Returns empty (island left intact) only when nothing useful can be
// done: total already fits one chunk, or it collapses to a single chunk.
std::vector<ExtrusionEntityCollection*> subdivide_infill_island(
    ExtrusionEntityCollection& island, double target_chunk_volume)
{
    std::vector<ExtrusionEntityCollection*> chunks;
    if (target_chunk_volume <= 0.0)
        return chunks;
    if (island.total_volume() <= target_chunk_volume)
        return chunks;

    // Build the working piece list. Oversized ExtrusionPaths are cut into
    // sub-paths (originals deleted); everything else is transferred as-is.
    std::vector<ExtrusionEntity*> pieces;
    pieces.reserve(island.entities.size());
    for (ExtrusionEntity* sub : island.entities) {
        auto* path = dynamic_cast<ExtrusionPath*>(sub);
        if (path != nullptr && path->total_volume() > target_chunk_volume) {
            split_path_by_volume(*path, target_chunk_volume, pieces);
            delete path;                       // replaced by the pieces above
        } else {
            pieces.push_back(sub);             // transfer ownership unchanged
        }
    }
    // Source pointers are now either deleted (split originals) or owned by
    // `pieces`; drop them without deleting.
    island.entities.clear();

    // Group the (now fine-grained) pieces into chunks of about target volume.
    auto open_chunk = [&island]() {
        auto* c = new ExtrusionEntityCollection();
        c->no_sort = island.no_sort;
        return c;
    };

    ExtrusionEntityCollection* current = open_chunk();
    double current_vol = 0.0;
    for (ExtrusionEntity* sub : pieces) {
        const double v = sub->total_volume();
        if (current_vol > 0.0 && current_vol + v > target_chunk_volume) {
            chunks.push_back(current);
            current = open_chunk();
            current_vol = 0.0;
        }
        current->entities.push_back(sub); // transfer ownership, no clone
        current_vol += v;
    }
    if (current->entities.empty())
        delete current;
    else
        chunks.push_back(current);

    // No benefit if it all collapsed into one chunk: hand the entities back so
    // the caller keeps the original island, and report "nothing done".
    if (chunks.size() <= 1) {
        if (chunks.size() == 1) {
            island.entities = std::move(chunks.front()->entities);
            delete chunks.front();
            chunks.clear();
        }
    }
    return chunks;
}

// Ordered per-transition purge demand for the greedy router simulator. Unlike a
// per-destination sum, this preserves the exact transition sequence the router
// processes plus every transition's raw matrix demand — including tiny-matrix
// transitions whose headroom is zero, which still consume baseline infill in the
// router's greedy pass.
//   demand   = matrix * flush_multiplier  (router's volume_to_wipe, pre-grab)
//   headroom = max(0, demand - grab - floor)  (useful routable for net-zero)
struct OrderedTx2 {
    unsigned int dest_fil;
    double       demand;
    double       headroom;
};
using LayerTxList  = std::vector<OrderedTx2>;
using OrderedTxMap = std::map<double, LayerTxList>;

void build_ordered_transitions_type1(const Print& print,
                                     const ToolOrdering& tool_ordering,
                                     OrderedTxMap& out)
{
    const PrintConfig& cfg = print.config();
    const size_t nozzle_nums         = cfg.nozzle_diameter.values.size();
    const size_t number_of_extruders = cfg.filament_colour.values.size();
    if (nozzle_nums == 0 || number_of_extruders == 0 || tool_ordering.empty()) return;

    std::vector<std::vector<std::vector<float>>> multi_extruder_flush;
    multi_extruder_flush.reserve(nozzle_nums);
    for (size_t nozzle_id = 0; nozzle_id < nozzle_nums; ++nozzle_id) {
        auto flat_d = get_flush_volumes_matrix(cfg.flush_volumes_matrix.values, nozzle_id, nozzle_nums);
        std::vector<float> flat_f(flat_d.begin(), flat_d.end());
        multi_extruder_flush.emplace_back(reshape_wipe_volumes(flat_f, number_of_extruders));
    }

    const std::vector<int> filament_maps = print.get_filament_maps();
    if (filament_maps.empty()) return;

    unsigned int current_filament_id = tool_ordering.first_extruder();
    std::vector<unsigned int> nozzle_cur_filament_ids(nozzle_nums, (unsigned int)(-1));
    if (current_filament_id < filament_maps.size()) {
        int initial_nozzle_id = filament_maps[current_filament_id] - 1;
        if (initial_nozzle_id >= 0 && size_t(initial_nozzle_id) < nozzle_nums)
            nozzle_cur_filament_ids[initial_nozzle_id] = current_filament_id;
    }

    for (const LayerTools& lt : tool_ordering) {
        if (!lt.has_wipe_tower) continue;
        LayerTxList txs;
        for (unsigned int filament_id : lt.extruders) {
            if (filament_id == current_filament_id) continue;
            if (filament_id >= filament_maps.size()) continue;
            int nozzle_id = filament_maps[filament_id] - 1;
            if (nozzle_id < 0 || size_t(nozzle_id) >= nozzle_nums) continue;

            unsigned int pre_filament_id = nozzle_cur_filament_ids[nozzle_id];
            if (pre_filament_id != (unsigned int)(-1) && pre_filament_id != filament_id
                && transition_is_absorbable(cfg, pre_filament_id, filament_id))
            {
                double demand = double(multi_extruder_flush[nozzle_id][pre_filament_id][filament_id])
                              * double(cfg.flush_multiplier.get_at(nozzle_id));
                if (demand > 0.0) {
                    const double grab = double(cfg.grab_length.get_at(nozzle_id)) * 2.4;
                    const double headroom = std::max(0.0, demand - grab - double(k_min_purge_volume_type1));
                    txs.push_back({filament_id, demand, headroom});
                }
            }
            current_filament_id = filament_id;
            nozzle_cur_filament_ids[nozzle_id] = filament_id;
        }
        if (!txs.empty())
            out[lt.print_z] = std::move(txs);
    }
}

void build_ordered_transitions_type2(const Print& print,
                                     const ToolOrdering& tool_ordering,
                                     OrderedTxMap& out)
{
    const PrintConfig& cfg = print.config();
    if (!cfg.purge_in_prime_tower || !cfg.single_extruder_multi_material || tool_ordering.empty()) return;
    const size_t number_of_extruders = cfg.filament_colour.values.size();
    if (number_of_extruders == 0) return;

    std::vector<float> flush_matrix(cfg.flush_volumes_matrix.values.begin(),
                                    cfg.flush_volumes_matrix.values.end());
    if (flush_matrix.size() < number_of_extruders * number_of_extruders) return;
    auto wipe_volumes = reshape_wipe_volumes(flush_matrix, number_of_extruders);

    const auto& all_extruders = tool_ordering.all_extruders();
    if (all_extruders.empty()) return;
    unsigned int current_extruder_id = all_extruders.back();

    for (const LayerTools& lt : tool_ordering) {
        if (!lt.has_wipe_tower) continue;
        const bool first_layer = (&lt == &tool_ordering.front());
        LayerTxList txs;
        for (unsigned int extruder_id : lt.extruders) {
            const bool is_transition =
                (first_layer && extruder_id == all_extruders.back()) || extruder_id != current_extruder_id;
            if (is_transition && transition_is_absorbable(cfg, current_extruder_id, extruder_id)) {
                double demand = double(wipe_volumes[current_extruder_id][extruder_id])
                              * double(cfg.flush_multiplier.get_at(0));
                if (demand > 0.0) {
                    // The minimal-purge floor always reaches the tower (the planner
                    // re-adds it after routing), so it is carved out of the bump
                    // budget: infill is only ever bumped to demand - floor.
                    const double floor2 = double(cfg.filament_minimal_purge_on_wipe_tower.get_at(extruder_id));
                    const double headroom = std::max(0.0, demand - floor2);
                    txs.push_back({extruder_id, demand, headroom});
                }
            }
            current_extruder_id = extruder_id;
        }
        if (!txs.empty())
            out[lt.print_z] = std::move(txs);
    }
}

// Quantize print_z to 1 um so per-z values across objects key cleanly even when
// float math leaves them not bit-identical between objects.
inline int64_t z_key(double print_z) {
    return int64_t(std::llround(print_z * 1000.0));
}

// Per-region bump capacity and routing state for the greedy allocator. Each
// region's V100/baseline is computed independently; allocation per layer mirrors
// the router's per-transition sequential consumption so only regions whose bumped
// infill the router will physically consume get bumped.
struct RegionAlloc {
    const PrintObject* obj;          // owning (possibly shadow) object; key for result map
    size_t             region_idx;   // index into src_layer->regions()
    unsigned int       wall_fil;     // 0-based, via lt.wall_extruder_id(region)
    double             v100;         // mm3 if filled to 100% (x instances)
    double             baseline;     // mm3 at current density (x instances)
    bool               is_infill_first; // region.config().is_infill_first
    bool               eligible;     // density == global_default -> bumpable
    double             bump;          // mm3 added above baseline (eligible only)
};

// Walk all opted-in objects at the given print_z and return per-region
// capacities. Uses the shared object's layers (shadow objects have empty layers
// during this pass) but tags each entry with the *caller* object so the caller
// can look up its allocations afterwards.
//
// Includes EVERY sparse-infill region, not just bump-eligible ones: the router
// consumes baseline infill from custom-density and untouched regions too, and the
// greedy sim must account for that consumption. Non-eligible regions carry
// eligible=false (baseline counted, never bumped).
std::vector<RegionAlloc> per_region_capacities(
    const std::vector<const PrintObject*>& enabled_objects,
    double print_z,
    float global_default,
    const LayerTools& lt)
{
    std::vector<RegionAlloc> out;
    for (const PrintObject* obj : enabled_objects) {
        const PrintObject* src = obj->get_shared_object() ? obj->get_shared_object() : obj;
        const Layer* src_layer = src->get_layer_at_printz(print_z, EPSILON);
        if (!src_layer) continue;
        const double n_inst = double(std::max<size_t>(obj->instances().size(), 1));
        const double layer_h = double(src_layer->height);
        const auto& regions = src_layer->regions();
        for (size_t ridx = 0; ridx < regions.size(); ++ridx) {
            const LayerRegion* lr = regions[ridx];
            const PrintRegionConfig& rc = lr->region().config();
            const float region_density = float(rc.sparse_infill_density);
            if (region_density <= 0.f) continue;  // no sparse infill at all
            double sparse_area_mm2 = 0.0;
            for (const Surface& surf : lr->fill_surfaces.surfaces) {
                if (surf.surface_type == stInternal)
                    sparse_area_mm2 += double(surf.area()) * SCALING_FACTOR * SCALING_FACTOR;
            }
            if (sparse_area_mm2 <= 0.0) continue;
            const double v100     = sparse_area_mm2 * layer_h * n_inst;
            const double baseline = v100 * double(region_density) / 100.0;
            const bool   eligible = (region_density == global_default);
            const bool   inf_first = rc.is_infill_first;
            const unsigned int wall_fil = lt.wall_extruder_id(lr->region());
            out.push_back({obj, ridx, wall_fil, v100, baseline, inf_first, eligible, 0.0});
        }
    }
    // Match mark_wiping_extrusions iteration order: dedicated-for-wiping objects
    // first (flush_into_objects=true), then by object id, then regions in order.
    std::sort(out.begin(), out.end(), [](const RegionAlloc& a, const RegionAlloc& b) {
        const bool a_ded = a.obj->config().flush_into_objects;
        const bool b_ded = b.obj->config().flush_into_objects;
        if (a_ded != b_ded) return a_ded;
        if (a.obj->id() != b.obj->id()) return a.obj->id() < b.obj->id();
        return a.region_idx < b.region_idx;
    });
    return out;
}

// Tower-side closed loop (Type2 / WipeTower2 only). Generate a throwaway wipe
// tower from explicit per-(layer, destination) wipe volumes and return each
// layer's realized extrusion volume (mm3: wipe + walls + ramming — everything the
// tower prints at that z). The router-side measurement credits absorption
// mm3-for-mm3, but the realized tower has per-layer minimum geometry (first wipe
// line, whole-line depth quantisation, stabilisation) plus the structural support
// envelope, so the marginal tower saving per absorbed mm3 can be below 1 on a
// small tower. Because this runs WipeTower2 itself rather than a model of it, the
// volumes include every such effect exactly. Returns false on Type1 (BBL) towers.
bool dip_tower_layer_volumes(
    const Print& print,
    const std::map<std::pair<int64_t, unsigned int>, float>& vol_by_z_dest,
    std::map<int64_t, double>& out)
{
    out.clear();
    if (print.wipe_tower_type() != WipeTowerType::Type2) return false;
    const PrintConfig& cfg = print.config();
    const size_t n_ext = cfg.filament_colour.values.size();
    if (n_ext == 0) return false;

    ToolOrdering to(print, (unsigned int) -1, /*prime_multi_material=*/true,
                    /*predict_no_entities=*/true);
    to.sort_and_build_data(print, (unsigned int) -1, true);
    if (!to.has_wipe_tower() || to.all_extruders().empty()) return false;

    std::vector<float> flush_matrix(cfg.flush_volumes_matrix.values.begin(),
                                    cfg.flush_volumes_matrix.values.end());
    if (flush_matrix.size() < n_ext * n_ext) return false;
    auto wipe_volumes = reshape_wipe_volumes(flush_matrix, n_ext);

    WipeTower2 wt(cfg, print.default_region_config(), print.get_plate_index(),
                  print.get_plate_origin(), wipe_volumes, to.first_extruder());
    for (size_t i = 0; i < n_ext; ++i)
        wt.set_extruder(i, cfg);
    // Prime once so internal state matches the real tower (priming TCRs discarded).
    wt.prime((float) print.skirt_first_layer_height(), to.all_extruders(), false);

    const bool  purge_in_tower = cfg.purge_in_prime_tower && cfg.single_extruder_multi_material;
    const float flush_mult     = (float) cfg.flush_multiplier.get_at(0);
    unsigned int current = to.all_extruders().back();
    for (LayerTools& lt : to.layer_tools()) {
        if (!lt.has_wipe_tower) continue;
        const bool first_layer = (&lt == &to.front());
        wt.plan_toolchange((float) lt.print_z, (float) lt.wipe_tower_layer_height,
                           current, current, false);
        for (unsigned int e : lt.extruders) {
            if ((first_layer && e == to.all_extruders().back()) || e != current) {
                float vol = (float) cfg.prime_volume;
                if (purge_in_tower) {
                    auto it = vol_by_z_dest.find(std::make_pair(z_key(lt.print_z), e));
                    vol = (it != vol_by_z_dest.end())
                        ? it->second
                        : wipe_volumes[current][e] * flush_mult;  // unmeasured: raw demand
                }
                wt.plan_toolchange((float) lt.print_z, (float) lt.wipe_tower_layer_height,
                                   current, e, vol);
                current = e;
            }
        }
        if (&lt == &to.back() || (&lt + 1)->wipe_tower_partitions == 0)
            break;
    }

    std::vector<std::vector<WipeTower::ToolChangeResult>> tcrs;
    tcrs.reserve(to.layer_tools().size());
    wt.generate(tcrs);
    if (tcrs.empty()) return false;

    // Realized volume per layer: sum over extrusion segments of len * width * h.
    // (Approximate vs the E-axis, but identical between the two generations the
    // caller compares, so the per-layer deltas are exact for clamping purposes.)
    for (const auto& layer : tcrs)
        for (const WipeTower::ToolChangeResult& tcr : layer) {
            double vol = 0.0;
            const auto& ex = tcr.extrusions;
            for (size_t i = 1; i < ex.size(); ++i)
                if (ex[i].width > 0.f)
                    vol += double((ex[i].pos - ex[i - 1].pos).norm())
                         * double(ex[i].width) * double(tcr.layer_height);
            out[z_key(double(tcr.print_z))] += vol;
        }
    return !out.empty();
}

} // anonymous namespace

DynamicInfillPurge::ObjectOverrides
DynamicInfillPurge::compute_plate_overrides(const Print& print)
{
    ObjectOverrides result;

    std::vector<const PrintObject*> enabled_objects;
    for (const PrintObject* obj : print.objects())
        if (obj->config().enable_dynamic_infill_purge)
            enabled_objects.push_back(obj);
    if (enabled_objects.empty()) return result;

    const float global_default =
        float(print.default_region_config().sparse_infill_density);

    // Build a predict-mode ToolOrdering: it uses fill_surfaces in place of
    // fills.entities (which do not exist yet between prepare_infill and
    // make_fills) and skips the side-effect marking, so transitions land on
    // exactly the layers the real wipe-tower planner will visit.
    const bool is_type2 = print.wipe_tower_type() == WipeTowerType::Type2;
    ToolOrdering predict_to(print, (unsigned int) -1, is_type2, /*predict_no_entities=*/true);
    predict_to.sort_and_build_data(print, (unsigned int) -1, is_type2);
    std::unordered_map<int64_t, const LayerTools*> lt_by_z;
    for (const LayerTools& lt : predict_to)
        lt_by_z[z_key(lt.print_z)] = &lt;

    OrderedTxMap tx_by_z;
    if (is_type2) build_ordered_transitions_type2(print, predict_to, tx_by_z);
    else          build_ordered_transitions_type1(print, predict_to, tx_by_z);

    // Phase 1: per-z greedy router simulation, keyed by z so all objects sharing
    // a layer cooperate on the same allocation decision.
    //
    // For each transition in router order the router grabs every reachable,
    // not-yet-consumed sparse-infill region until its matrix demand is met. We
    // mirror the exact reachability gate: an infill-first region has no order
    // constraint; otherwise its wall_filament must precede the destination. A
    // region's infill is shared across transitions, so we draw each region DOWN
    // cumulatively rather than locking it wholesale to the first transition that
    // touches it (which would let a tiny early transition starve a later, larger
    // one). Each transition first absorbs into remaining baseline (free), then
    // bumps for still-unmet demand within its net-zero budget (headroom) and the
    // region's physical capacity. Net zero by construction.
    std::unordered_map<int64_t, std::vector<RegionAlloc>> alloc_by_z;
    double total_bump_mm3 = 0.0;
    for (const auto& kv : tx_by_z) {
        const double print_z = kv.first;
        const LayerTxList& txs = kv.second;
        const int64_t k = z_key(print_z);

        auto lt_it = lt_by_z.find(k);
        if (lt_it == lt_by_z.end()) continue;
        const LayerTools& lt = *lt_it->second;

        auto regions = per_region_capacities(enabled_objects, print_z, global_default, lt);
        if (regions.empty()) continue;

        std::vector<double> baseline_left(regions.size());
        std::vector<double> bump_left(regions.size(), 0.0);
        for (size_t i = 0; i < regions.size(); ++i) {
            baseline_left[i] = regions[i].baseline;
            if (regions[i].eligible) {
                const float obj_max_d = std::max(global_default,
                    std::min(100.f, float(regions[i].obj->config().dynamic_infill_purge_density_max)));
                bump_left[i] = regions[i].v100 * double(obj_max_d - global_default) / 100.0;
            }
        }
        for (const OrderedTx2& tx : txs) {
            double demand = tx.demand;
            double useful = tx.headroom;
            for (size_t i = 0; i < regions.size(); ++i) {
                if (demand <= 0.0 || useful <= 0.0) break;
                RegionAlloc& r = regions[i];
                const bool reachable = r.is_infill_first
                                     || lt.is_extruder_order(r.wall_fil, tx.dest_fil);
                if (!reachable) continue;
                // 1) Absorb into remaining baseline first (free, no new material).
                const double take_base = std::min(baseline_left[i], std::min(demand, useful));
                if (take_base > 0.0) {
                    baseline_left[i] -= take_base;
                    demand           -= take_base;
                    useful           -= take_base;
                }
                // 2) Bump for still-unmet demand, within net-zero budget + capacity.
                if (demand > 0.0 && useful > 0.0 && bump_left[i] > 0.0) {
                    const double add = std::min(bump_left[i], std::min(demand, useful));
                    if (add > 0.0) {
                        bump_left[i] -= add;
                        r.bump       += add;   // accumulate across transitions
                        demand       -= add;
                        useful       -= add;
                        total_bump_mm3 += add;
                    }
                }
            }
        }
        alloc_by_z[k] = std::move(regions);
    }

    // Phase 2: walk each object's own layers, look up allocations by z, and write
    // per-region density overrides (only this object's allocations).
    for (const PrintObject* obj : enabled_objects) {
        const float max_d = std::max(global_default,
            std::min(100.f, float(obj->config().dynamic_infill_purge_density_max)));

        DensityOverrides overrides;
        for (const Layer* layer : obj->layers()) {
            const int64_t k = z_key(layer->print_z);
            auto a_it = alloc_by_z.find(k);
            if (a_it == alloc_by_z.end()) continue;

            RegionDensityMap layer_overrides;
            for (const RegionAlloc& r : a_it->second) {
                if (r.obj != obj) continue;
                if (r.bump <= 0.0) continue;
                double density_pct = (r.baseline + r.bump) / r.v100 * 100.0;
                density_pct = std::min(double(max_d), std::max(double(global_default), density_pct));
                layer_overrides[r.region_idx] = float(density_pct);
            }
            if (!layer_overrides.empty())
                overrides[layer->id()] = std::move(layer_overrides);
        }
        result[obj] = std::move(overrides);
    }

    BOOST_LOG_TRIVIAL(debug) << "DynamicInfillPurge: allocated " << total_bump_mm3
                             << " mm3 of infill bump across " << enabled_objects.size()
                             << " object(s)";
    return result;
}

DynamicInfillPurge::MeasuredRouting
DynamicInfillPurge::measure_routing(const Print& print,
                                    DynamicInfillPurge::ClaimedPerRegion* claimed_out)
{
    // Sum, per (object, layer, region), the volume of infill entities the
    // throwaway router has marked for wiping at this LayerTools. Single-copy
    // volumes (copy 0), matching how volume_to_wipe is consumed.
    auto accumulate_claimed = [&](LayerTools& lt) {
        if (!claimed_out) return;
        for (const PrintObject* obj : print.objects()) {
            if (!obj->config().enable_dynamic_infill_purge) continue;
            const Layer* layer = obj->get_layer_at_printz(lt.print_z, EPSILON);
            if (!layer) continue;
            const auto& regions = layer->regions();
            for (int ridx = 0; ridx < int(regions.size()); ++ridx) {
                double sum = 0.0;
                for (const ExtrusionEntity* ee : regions[ridx]->fills.entities)
                    if (lt.wiping_extrusions().dip_entity_claimed(ee, obj))
                        sum += ee->total_volume();
                if (sum > 0.0)
                    (*claimed_out)[std::make_tuple(obj, z_key(lt.print_z), ridx)] += sum;
            }
        }
    };

    MeasuredRouting out;
    const PrintConfig& cfg = print.config();
    const bool is_type2 = print.wipe_tower_type() == WipeTowerType::Type2;

    // Build a throwaway ToolOrdering from the CURRENT fills (not predict mode):
    // mark_wiping_extrusions needs real fills.entities to route. Overrides land in
    // this local ToolOrdering's WipingExtrusions, never on the entities, so this
    // is non-destructive and the real _make_wipe_tower later is unaffected.
    ToolOrdering to(print, (unsigned int) -1, is_type2);
    to.sort_and_build_data(print, (unsigned int) -1, is_type2);
    if (!to.has_wipe_tower()) return out;

    const size_t number_of_extruders = cfg.filament_colour.values.size();
    if (number_of_extruders == 0) return out;

    if (!is_type2) {
        // Mirror the _make_wipe_tower Type1 loop.
        const size_t nozzle_nums = cfg.nozzle_diameter.values.size();
        if (nozzle_nums == 0) return out;

        std::vector<std::vector<std::vector<float>>> multi_extruder_flush;
        multi_extruder_flush.reserve(nozzle_nums);
        for (size_t nozzle_id = 0; nozzle_id < nozzle_nums; ++nozzle_id) {
            auto flat_d = get_flush_volumes_matrix(cfg.flush_volumes_matrix.values, nozzle_id, nozzle_nums);
            std::vector<float> flat_f(flat_d.begin(), flat_d.end());
            multi_extruder_flush.emplace_back(reshape_wipe_volumes(flat_f, number_of_extruders));
        }

        const std::vector<int> filament_maps = print.get_filament_maps();
        if (filament_maps.empty()) return out;

        unsigned int current_filament_id = to.first_extruder();
        std::vector<unsigned int> nozzle_cur_filament_ids(nozzle_nums, (unsigned int)(-1));
        if (current_filament_id < filament_maps.size()) {
            int initial_nozzle_id = filament_maps[current_filament_id] - 1;
            if (initial_nozzle_id >= 0 && size_t(initial_nozzle_id) < nozzle_nums)
                nozzle_cur_filament_ids[initial_nozzle_id] = current_filament_id;
        }

        for (LayerTools& lt : to.layer_tools()) {
            if (!lt.has_wipe_tower) continue;
            std::vector<MeasuredTx> txs;
            for (unsigned int filament_id : lt.extruders) {
                if (filament_id == current_filament_id) continue;
                if (filament_id >= filament_maps.size()) continue;
                int nozzle_id = filament_maps[filament_id] - 1;
                if (nozzle_id < 0 || size_t(nozzle_id) >= nozzle_nums) continue;

                unsigned int pre_filament_id = nozzle_cur_filament_ids[nozzle_id];
                if (pre_filament_id != (unsigned int)(-1) && pre_filament_id != filament_id) {
                    float volume_in = multi_extruder_flush[nozzle_id][pre_filament_id][filament_id]
                                    * float(cfg.flush_multiplier.get_at(nozzle_id));
                    float volume_out = lt.wiping_extrusions().mark_wiping_extrusions(
                        print, current_filament_id, filament_id, volume_in);
                    txs.push_back({filament_id, volume_in, volume_out});
                }
                current_filament_id = filament_id;
                nozzle_cur_filament_ids[nozzle_id] = filament_id;
            }
            accumulate_claimed(lt);   // capture per-region claims before order fixup
            lt.wiping_extrusions().ensure_perimeters_infills_order(print);
            if (!txs.empty()) out[lt.print_z] = std::move(txs);
        }
    } else {
        // Mirror the _make_wipe_tower Type2 loop.
        if (!cfg.purge_in_prime_tower || !cfg.single_extruder_multi_material) return out;

        std::vector<float> flush_matrix(cfg.flush_volumes_matrix.values.begin(),
                                        cfg.flush_volumes_matrix.values.end());
        if (flush_matrix.size() < number_of_extruders * number_of_extruders) return out;
        auto wipe_volumes = reshape_wipe_volumes(flush_matrix, number_of_extruders);

        const auto& all_extruders = to.all_extruders();
        if (all_extruders.empty()) return out;
        unsigned int current_extruder_id = all_extruders.back();

        for (LayerTools& lt : to.layer_tools()) {
            if (!lt.has_wipe_tower) continue;
            const bool first_layer = (&lt == &to.front());
            std::vector<MeasuredTx> txs;
            for (unsigned int extruder_id : lt.extruders) {
                const bool is_transition =
                    (first_layer && extruder_id == all_extruders.back()) || extruder_id != current_extruder_id;
                if (is_transition) {
                    // Mirror the real planner's floor handling exactly: the
                    // minimal-purge floor is carved out before routing and re-added
                    // onto the tower after, so volume_out is what the tower sees.
                    const float floor = float(cfg.filament_minimal_purge_on_wipe_tower.get_at(extruder_id));
                    float volume_in = wipe_volumes[current_extruder_id][extruder_id]
                                    * float(cfg.flush_multiplier.get_at(0));
                    volume_in -= floor; if (volume_in < 0.f) volume_in = 0.f;
                    float volume_out = lt.wiping_extrusions().mark_wiping_extrusions(
                        print, current_extruder_id, extruder_id, volume_in);
                    volume_out += floor;
                    txs.push_back({extruder_id, volume_in, volume_out});
                }
                current_extruder_id = extruder_id;
            }
            accumulate_claimed(lt);   // capture per-region claims before order fixup
            lt.wiping_extrusions().ensure_perimeters_infills_order(print);
            if (!txs.empty()) out[lt.print_z] = std::move(txs);
        }
    }

    return out;
}

DynamicInfillPurge::ObjectOverrides
DynamicInfillPurge::compute_trim_overrides(const Print& print,
                                           const ClaimedPerRegion& claimed,
                                           std::map<int64_t, double>* bump_by_z)
{
    ObjectOverrides result;
    const float global_default = float(print.default_region_config().sparse_infill_density);

    for (const PrintObject* obj : print.objects()) {
        if (!obj->config().enable_dynamic_infill_purge) continue;
        const float max_d = std::max(global_default,
            std::min(100.f, float(obj->config().dynamic_infill_purge_density_max)));
        const double n_inst = double(std::max<size_t>(obj->instances().size(), 1));

        DensityOverrides overrides;
        for (const Layer* layer : obj->layers()) {
            const int64_t k = z_key(layer->print_z);
            const double  layer_h = double(layer->height);
            RegionDensityMap rdm;
            const auto& regions = layer->regions();
            for (int ridx = 0; ridx < int(regions.size()); ++ridx) {
                const LayerRegion* lr = regions[ridx];
                // Only eligible regions (still at the global default) can carry a
                // bump; others keep their user-set density untouched.
                if (float(lr->region().config().sparse_infill_density) != global_default)
                    continue;
                double area_mm2 = 0.0;
                for (const Surface& s : lr->fill_surfaces.surfaces)
                    if (s.surface_type == stInternal)
                        area_mm2 += double(s.area()) * SCALING_FACTOR * SCALING_FACTOR;
                if (area_mm2 <= 0.0) continue;

                const double v100         = area_mm2 * layer_h;            // single copy
                const double baseline_vol = v100 * double(global_default) / 100.0;
                auto it = claimed.find(std::make_tuple(obj, k, ridx));
                const double claimed_vol  = (it == claimed.end()) ? 0.0 : it->second;

                // Keep infill only up to what the router actually claims; never
                // below the model's baseline density. Orphan bump (claimed below
                // baseline) trims away entirely -> no override emitted.
                const double target_vol = std::max(baseline_vol, claimed_vol);
                double density = target_vol / v100 * 100.0;
                density = std::min(double(max_d), std::max(double(global_default), density));
                if (density > double(global_default) + 1e-6) {
                    rdm[size_t(ridx)] = float(density);
                    if (bump_by_z)   // plate-volume bump credit for the tower-side clamp
                        (*bump_by_z)[k] += (v100 * density / 100.0 - baseline_vol) * n_inst;
                }
            }
            if (!rdm.empty()) overrides[layer->id()] = std::move(rdm);
        }
        if (!overrides.empty()) result[obj] = std::move(overrides);
    }
    return result;
}

void DynamicInfillPurge::apply_island_subdivision(const std::vector<PrintObject*>& objects)
{
    for (PrintObject* obj : objects) {
        if (!obj->config().enable_dynamic_infill_purge) continue;

        for (Layer* layer : obj->layers()) {            // shadow objects: empty -> no-op
            for (int ridx = 0; ridx < int(layer->region_count()); ++ridx) {
                LayerRegion* lr = layer->get_region(ridx);
                if (lr == nullptr) continue;

                ExtrusionEntitiesPtr new_entities;
                new_entities.reserve(lr->fills.entities.size());
                bool changed = false;
                for (ExtrusionEntity* ee : lr->fills.entities) {
                    auto* island = dynamic_cast<ExtrusionEntityCollection*>(ee);
                    if (island != nullptr
                        && island->role() == erInternalInfill
                        && island->total_volume() > k_dip_max_island_volume_mm3) {
                        auto chunks = subdivide_infill_island(*island, k_dip_max_island_volume_mm3);
                        if (!chunks.empty()) {
                            for (ExtrusionEntityCollection* c : chunks)
                                new_entities.push_back(c);
                            delete island;          // emptied by subdivide; safe
                            changed = true;
                            continue;
                        }
                    }
                    new_entities.push_back(ee);     // unchanged (kept, not split)
                }
                if (changed)
                    lr->fills.entities = std::move(new_entities);
            }
        }
    }
}

void DynamicInfillPurge::apply_closed_loop_trim(const std::vector<PrintObject*>& objects,
                                                const Print& print)
{
    // Nothing to measure or trim if no object opted in — skip the throwaway
    // tower-routing pass entirely (keeps non-feature slices free of overhead).
    bool any_enabled = false;
    for (const PrintObject* obj : objects)
        if (obj->config().enable_dynamic_infill_purge) { any_enabled = true; break; }
    if (!any_enabled) return;

    // Measure what the real router claims per region on the CURRENT (bumped,
    // subdivided) fills. `measured` also carries per-(z, dest) leftover
    // (volume_out) — exactly the wipe volumes the real tower will be fed.
    ClaimedPerRegion claimed;
    MeasuredRouting measured = measure_routing(print, &claimed);

    std::map<int64_t, double> bump_by_z;
    ObjectOverrides trimmed = compute_trim_overrides(print, claimed, &bump_by_z);

    // Tower-side closed loop (Type2 only): generate the throwaway tower twice —
    // once fed the measured leftovers (what the real tower will print) and once
    // with each layer's bump-credited absorption returned to the tower (the
    // no-bump counterfactual). The per-layer realized volume difference is the
    // bump's real tower saving; scale each layer's kept bump down to it so no
    // model material is spent on savings the tower geometry will not realize
    // (per-layer minimums + support envelope clip them on small towers).
    std::map<int64_t, double> ratio_by_z;
    if (!bump_by_z.empty()) {
        std::map<std::pair<int64_t, unsigned int>, float> vol_now, vol_plus;
        for (const auto& kv : measured) {
            const int64_t zk = z_key(kv.first);
            double absorbed_z = 0.0;
            for (const MeasuredTx& tx : kv.second)
                absorbed_z += double(std::max(0.f, tx.volume_in - tx.volume_out));
            auto bit = bump_by_z.find(zk);
            const double credit = (bit == bump_by_z.end()) ? 0.0 : bit->second;
            for (const MeasuredTx& tx : kv.second) {
                const auto key = std::make_pair(zk, tx.dest_fil);
                const double tx_abs = double(std::max(0.f, tx.volume_in - tx.volume_out));
                // Return this transition's share of the bump credit to the tower,
                // never more than it absorbed in the first place.
                const double share = absorbed_z > 0.0
                    ? std::min(credit * tx_abs / absorbed_z, tx_abs) : 0.0;
                vol_now[key]  += tx.volume_out;
                vol_plus[key] += tx.volume_out + float(share);
            }
        }
        std::map<int64_t, double> t_now, t_plus;
        if (dip_tower_layer_volumes(print, vol_now, t_now)
         && dip_tower_layer_volumes(print, vol_plus, t_plus)) {
            for (const auto& kv : bump_by_z) {
                const int64_t zk     = kv.first;
                const double  credit = kv.second;
                if (credit <= 1e-6) continue;
                const double saving = std::max(0.0, t_plus[zk] - t_now[zk]);
                ratio_by_z[zk] = std::min(1.0, saving / credit);
            }
        }
    }

    const float global_default = float(print.default_region_config().sparse_infill_density);
    for (PrintObject* obj : objects) {
        if (!obj->config().enable_dynamic_infill_purge) continue;
        auto it = trimmed.find(obj);
        if (it != trimmed.end() && !ratio_by_z.empty()) {
            // Apply the per-layer tower clamp to this object's kept overrides.
            for (const Layer* layer : obj->layers()) {
                auto oit = it->second.find(layer->id());
                if (oit == it->second.end()) continue;
                auto rit = ratio_by_z.find(z_key(layer->print_z));
                if (rit == ratio_by_z.end() || rit->second >= 0.999) continue;
                for (auto rd = oit->second.begin(); rd != oit->second.end(); ) {
                    const float d = global_default
                        + float((double(rd->second) - double(global_default)) * rit->second);
                    if (d > global_default + 1e-4f) { rd->second = d; ++rd; }
                    else                            { rd = oit->second.erase(rd); }
                }
                if (oit->second.empty()) it->second.erase(oit);
            }
        }
        if (it == trimmed.end() || it->second.empty())
            obj->clear_dynamic_purge_density_overrides();   // all bump was orphan -> baseline
        else
            obj->set_dynamic_purge_density_overrides(std::move(it->second));
        obj->make_fills_redo();                              // regenerate at trimmed density
    }

    // Re-subdivide the freshly re-filled islands so the real _make_wipe_tower
    // router sees the same fine-grained chunks the measurement was based on.
    apply_island_subdivision(objects);
}

} // namespace Slic3r
