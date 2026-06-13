#pragma once

// Dynamic infill purge: absorb multi-material purge into model sparse infill.
//
// On layers where a tool change occurs, the flush/wipe volume that would
// otherwise be deposited on the prime tower can instead be routed into the
// object's own sparse infill (via OrcaSlicer's existing flush_into_infill
// router). This raises the sparse-infill density of eligible regions just
// enough to absorb the predicted purge, then measures what the router actually
// claims and trims the rest back — so net material use stays at or below the
// no-feature baseline while the prime tower shrinks.
//
// Gated per object by `enable_dynamic_infill_purge` (default off); when off,
// output is byte-identical to baseline.

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Slic3r {

class Print;
class PrintObject;

class DynamicInfillPurge
{
public:
    // layer_id -> (region index within layer.regions() -> density % in [0, 100]).
    using RegionDensityMap = std::unordered_map<size_t, float>;
    using DensityOverrides = std::unordered_map<size_t, RegionDensityMap>;
    using ObjectOverrides  = std::unordered_map<const PrintObject*, DensityOverrides>;

    // Per-(layer, region) sparse-infill density bumps for every opted-in object,
    // sized so the wipe router can absorb each layer's predicted purge into
    // infill. The existing sparse_infill_density is the floor;
    // dynamic_infill_purge_density_max is the ceiling. Run between
    // prepare_infill() and make_fills().
    static ObjectOverrides compute_plate_overrides(const Print& print);

    // Subdivide oversized erInternalInfill islands into ~chunk-sized collections
    // so the wipe router can distribute one region's infill across several tool
    // changes (and so the closed-loop measurement tracks demand rather than
    // whole-island size). Geometry is byte-identical, only regrouped at
    // whole-path boundaries. Run after fills exist, before routing/measurement.
    static void apply_island_subdivision(const std::vector<PrintObject*>& objects);

    // Per-transition purge measured by running the real wipe router
    // (mark_wiping_extrusions) against the current fills on a throwaway
    // ToolOrdering. volume_in = demand fed to the router; volume_out = residual
    // still bound for the tower after infill absorption. Non-destructive — the
    // overrides live in the throwaway ToolOrdering, never on the fill entities.
    struct MeasuredTx {
        unsigned int dest_fil;
        float        volume_in;
        float        volume_out;
    };
    using MeasuredRouting = std::map<double, std::vector<MeasuredTx>>;  // key: print_z

    // Per-region volume the router actually claims for wiping (mm3, single copy),
    // keyed by (object, z quantised to um, region index) — the ground truth the
    // closed-loop trim sizes against.
    using ClaimedPerRegion =
        std::map<std::tuple<const PrintObject*, int64_t, int>, double>;
    static MeasuredRouting measure_routing(const Print& print,
                                           ClaimedPerRegion* claimed_out = nullptr);

    // Clamp every eligible region's infill to max(baseline, claimed): drop bumped
    // infill the router never claims (the orphan), keep what it does. When
    // `bump_by_z` is non-null it receives each layer's kept bump volume above
    // baseline (plate volume, instances included) for the tower-side clamp.
    static ObjectOverrides compute_trim_overrides(const Print& print,
                                                  const ClaimedPerRegion& claimed,
                                                  std::map<int64_t, double>* bump_by_z = nullptr);

    // Closed-loop trim: measure the post-bump router claims, trim bumped infill
    // back to what is actually absorbed, and additionally clamp each layer's bump
    // to the prime tower's REALIZED per-layer saving — per-layer minimum geometry
    // and the support envelope mean a small tower saves less than 1 mm3 per
    // absorbed mm3. Re-fills and re-subdivides. Run after the post-bump fill,
    // before _make_wipe_tower.
    static void apply_closed_loop_trim(const std::vector<PrintObject*>& objects,
                                       const Print& print);
};

} // namespace Slic3r
