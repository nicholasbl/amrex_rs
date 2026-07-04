use anyhow::{Context, Result, ensure};
use glam::{DVec3, IVec3, UVec3};

use crate::PlotFile;
use crate::sparse_amr::{SparseAmr, SparseAmrLevel, level_translation};
use crate::utility::{Aabb3u, SparseGrid3};

/// Cell-centered samples and eligible dual-lattice cube anchors for one AMR level.
///
/// All sparse grids use coordinates local to `index_origin`. Samples covered by
/// a finer level remain available; only `active_cubes` is masked.
pub(crate) struct DualGridLevel {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) physical_origin: DVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) samples: SparseGrid3<f32>,
    pub(crate) sampled_quantities: Vec<SparseGrid3<f32>>,
    pub(crate) active_cubes: SparseGrid3<()>,
}

/// Build the isosurface-specific dual lattice from reusable sparse AMR data.
///
/// Valid fine-level sample coverage masks eligible cube anchors on the
/// immediately coarser level. Fine and coarse meshes are not stitched here.
pub(super) fn load_dual_grid_levels(
    plot_file: &PlotFile,
    component: usize,
    sampled_components: &[usize],
) -> Result<Vec<DualGridLevel>> {
    ensure!(
        sampled_components.len() <= 2,
        "at most two sampled quantities are supported"
    );

    let component_ids = std::iter::once(component)
        .chain(sampled_components.iter().copied())
        .collect::<Vec<_>>();
    let sparse_amr = SparseAmr::load(plot_file, &component_ids)?;
    ensure!(
        sparse_amr.component_ids == component_ids,
        "sparse AMR component order changed while loading"
    );

    let mut levels: Vec<DualGridLevel> = Vec::with_capacity(sparse_amr.levels.len());
    for sparse_level in sparse_amr.levels {
        let SparseAmrLevel {
            level_index,
            index_origin,
            index_type,
            physical_origin,
            cell_size,
            components,
            valid_regions,
        } = sparse_level;
        ensure!(
            index_type == IVec3::ZERO,
            "isosurface extraction requires cell-centered data"
        );
        let mut components = components.into_iter();
        let samples = components
            .next()
            .context("surface component is absent from sparse AMR level")?;
        let active_cubes = build_active_dual_cubes(&samples, &valid_regions);
        let loaded = DualGridLevel {
            level_index,
            index_origin,
            physical_origin,
            cell_size,
            samples,
            sampled_quantities: components.collect(),
            active_cubes,
        };

        if let Some(coarse) = levels.last_mut() {
            let ratio = u32::try_from(plot_file.header().refinement_ratios[level_index - 1])
                .context("refinement ratio does not fit in u32")?;
            ensure!(ratio > 0, "refinement ratio must be nonzero");

            let translation = level_translation(coarse.index_origin, loaded.index_origin, ratio)?;
            coarse
                .active_cubes
                .mask_out_by_presence_scaled(&loaded.samples, ratio, translation);
        }

        levels.push(loaded);
    }

    Ok(levels)
}

pub(super) fn build_active_dual_cubes(
    samples: &SparseGrid3<f32>,
    sample_regions: &[Aabb3u],
) -> SparseGrid3<()> {
    let sample_bounds = samples.bounds();
    let cube_bounds = UVec3::new(
        sample_bounds.x.saturating_sub(1),
        sample_bounds.y.saturating_sub(1),
        sample_bounds.z.saturating_sub(1),
    );
    let mut candidates = SparseGrid3::with_chunk_capacity(cube_bounds, sample_regions.len());

    // A sample at p may contribute to cubes anchored at p and p - 1.
    for region in sample_regions {
        let candidate_region = Aabb3u::new(
            UVec3::new(
                region.min.x.saturating_sub(1),
                region.min.y.saturating_sub(1),
                region.min.z.saturating_sub(1),
            ),
            region.max,
        );
        candidates.fill_aabb(candidate_region, ());
    }

    let mut active = SparseGrid3::with_chunk_capacity(cube_bounds, candidates.chunk_count());
    let mut accessor = samples.accessor();
    candidates.for_each_present_in_aabb(candidates.bounds_aabb(), |anchor, ()| {
        let mut complete = true;
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    complete &= accessor.contains(anchor + UVec3::new(dx, dy, dz));
                }
            }
        }
        if complete {
            active.set(anchor, ());
        }
    });
    active
}
