use std::ops::RangeInclusive;

use anyhow::{Context, Result, bail, ensure};
use glam::{DVec3, I64Vec3, IVec3, U16Vec2, UVec3, Vec3};

use crate::{DataReader, IndexDomain, Level, PlotFile};

use super::sparse_grid3::{Aabb3u, SparseGrid3};

#[derive(Debug, Default)]
pub struct Surface {
    id: u32,
    value: f64,
}

#[derive(Debug)]
pub struct Sample {
    id: u32,
    min_value: RangeInclusive<f64>,
}

#[derive(Debug, Default)]
pub struct IsosurfaceOptions {
    surface: Surface,
    sample_quantity_ids: Vec<Sample>,
}

#[derive(Debug)]
pub struct Mesh3D {
    pub positions: Vec<Vertex3D>,
    pub faces: Vec<UVec3>,
}

#[derive(Debug)]
pub struct Vertex3D {
    position: Vec3,
    sampled_values: U16Vec2,
}

/// Cell-centered samples and eligible dual-lattice cube anchors for one AMR level.
///
/// Both sparse grids use coordinates local to `index_origin`. Samples covered by
/// a finer level remain available; only `active_cubes` is masked.
pub(crate) struct DualGridLevel {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) samples: SparseGrid3<f32>,
    pub(crate) active_cubes: SparseGrid3<()>,
}

/// Load one scalar component into a sparse dual lattice for every AMR level.
///
/// Levels are loaded from coarse to fine. Valid fine-level sample coverage masks
/// eligible cube anchors on the immediately coarser level. Fine and coarse meshes
/// are intentionally not stitched at this stage.
pub(crate) fn load_dual_grid_levels(
    plot_file: &PlotFile,
    component: usize,
) -> Result<Vec<DualGridLevel>> {
    ensure!(
        component < plot_file.variables().len(),
        "component index {component} is out of range"
    );

    let reader = DataReader::new(plot_file);
    let mut levels: Vec<DualGridLevel> = Vec::with_capacity(plot_file.header().finest_level + 1);

    for level in plot_file.levels() {
        let loaded = load_dual_grid_level(&reader, level, component)
            .with_context(|| format!("loading dual grid for level {}", level.index()))?;

        if let Some(coarse) = levels.last_mut() {
            let ratio = u32::try_from(plot_file.header().refinement_ratios[level.index() - 1])
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

fn load_dual_grid_level(
    reader: &DataReader<'_>,
    level: Level<'_>,
    component: usize,
) -> Result<DualGridLevel> {
    let domain = level.index_domain();
    ensure!(
        domain.index_type == IVec3::ZERO,
        "isosurface extraction requires cell-centered data"
    );

    let bounds = index_extent(domain)?;
    let mut samples = SparseGrid3::with_chunk_capacity(bounds, level.patch_count());
    let mut patch_aabbs = Vec::with_capacity(level.patch_count());

    for patch in level.patches()? {
        let patch_box = patch.index_box();
        ensure!(
            patch_box.index_type == IVec3::ZERO,
            "patch {} is not cell-centered",
            patch.index()
        );

        let local_aabb = local_aabb(patch_box, domain)?;
        let view = reader.component(patch, component)?;
        let value_count = local_aabb
            .volume_usize()
            .context("patch sample count does not fit in usize")?;
        let mut values = Vec::with_capacity(value_count);

        // Read only the valid index box. ComponentView may also contain ghost cells.
        for z in patch_box.min.z..=patch_box.max.z {
            for y in patch_box.min.y..=patch_box.max.y {
                for x in patch_box.min.x..=patch_box.max.x {
                    let index = IVec3::new(x, y, z);
                    let value = view.get(index).with_context(|| {
                        format!(
                            "valid index {index:?} is absent from patch {} component view",
                            patch.index()
                        )
                    })?;
                    values.push(value as f32);
                }
            }
        }

        samples
            .write_aabb_values(local_aabb, &values)
            .map_err(|error| anyhow::anyhow!("writing patch samples: {error:?}"))?;
        patch_aabbs.push(local_aabb);
    }

    let active_cubes = build_active_dual_cubes(&samples, &patch_aabbs);

    Ok(DualGridLevel {
        level_index: level.index(),
        index_origin: domain.min,
        cell_size: level.cell_size(),
        samples,
        active_cubes,
    })
}

fn build_active_dual_cubes(
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

fn index_extent(domain: &IndexDomain) -> Result<UVec3> {
    let extent = domain.max.as_i64vec3() - domain.min.as_i64vec3() + I64Vec3::ONE;
    ensure!(extent.cmpgt(I64Vec3::ZERO).all(), "index domain is empty");
    Ok(UVec3::new(
        u32::try_from(extent.x).context("x index extent does not fit in u32")?,
        u32::try_from(extent.y).context("y index extent does not fit in u32")?,
        u32::try_from(extent.z).context("z index extent does not fit in u32")?,
    ))
}

fn local_aabb(index_box: &IndexDomain, domain: &IndexDomain) -> Result<Aabb3u> {
    ensure!(
        index_box.min.cmpge(domain.min).all() && index_box.max.cmple(domain.max).all(),
        "patch index box lies outside its level domain"
    );
    let min = index_box.min.as_i64vec3() - domain.min.as_i64vec3();
    let max = index_box.max.as_i64vec3() - domain.min.as_i64vec3() + I64Vec3::ONE;
    Ok(Aabb3u::new(
        UVec3::new(
            u32::try_from(min.x)?,
            u32::try_from(min.y)?,
            u32::try_from(min.z)?,
        ),
        UVec3::new(
            u32::try_from(max.x)?,
            u32::try_from(max.y)?,
            u32::try_from(max.z)?,
        ),
    ))
}

fn level_translation(coarse_origin: IVec3, fine_origin: IVec3, scale: u32) -> Result<I64Vec3> {
    fn axis(coarse: i32, fine: i32, scale: u32) -> Result<i64> {
        let value = i128::from(coarse) * i128::from(scale) - i128::from(fine);
        i64::try_from(value).context("level-index translation does not fit in i64")
    }

    Ok(I64Vec3::new(
        axis(coarse_origin.x, fine_origin.x, scale)?,
        axis(coarse_origin.y, fine_origin.y, scale)?,
        axis(coarse_origin.z, fine_origin.z, scale)?,
    ))
}

fn isosurface(plot_file: &PlotFile, options: IsosurfaceOptions) -> Result<Mesh3D> {
    let component = usize::try_from(options.surface.id).context("surface id is too large")?;
    let levels = load_dual_grid_levels(plot_file, component)?;
    bail!(
        "RMT extraction is not implemented yet (loaded {} AMR levels for isovalue {})",
        levels.len(),
        options.surface.value
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_cubes_span_adjacent_sample_regions() {
        let mut samples = SparseGrid3::new(UVec3::new(3, 2, 2));
        let left = Aabb3u::new(UVec3::ZERO, UVec3::new(1, 2, 2));
        let right = Aabb3u::new(UVec3::new(1, 0, 0), UVec3::new(3, 2, 2));
        samples.fill_aabb(left, 1.0);
        samples.fill_aabb(right, 2.0);

        let active = build_active_dual_cubes(&samples, &[left, right]);
        assert_eq!(active.bounds(), UVec3::new(2, 1, 1));
        assert_eq!(active.present_voxel_count(), 2);
        assert!(active.contains(UVec3::new(0, 0, 0)));
        assert!(active.contains(UVec3::new(1, 0, 0)));
    }

    #[test]
    fn incomplete_dual_cube_is_not_active() {
        let mut samples = SparseGrid3::new(UVec3::new(2, 2, 2));
        let region = samples.bounds_aabb();
        samples.fill_aabb(region, 1.0);
        samples.clear(UVec3::new(1, 1, 1));

        let active = build_active_dual_cubes(&samples, &[region]);
        assert!(active.is_empty());
    }

    #[test]
    fn translation_relates_level_local_coordinates() {
        let translation =
            level_translation(IVec3::new(-4, 3, 10), IVec3::new(-7, 8, 20), 2).unwrap();
        assert_eq!(translation, I64Vec3::new(-1, -2, 0));
    }
}
