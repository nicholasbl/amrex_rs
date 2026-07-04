use anyhow::{Context, Result, ensure};
use glam::{DVec3, I64Vec3, IVec3, UVec3};

use crate::utility::{Aabb3u, SparseGrid3};
use crate::{DataReader, IndexDomain, Level, Patch, PlotFile};

/// Sparse, cell-centered component data for an AMR hierarchy.
///
/// Levels are ordered from coarse to fine. Component grids at every level use
/// the same order as `component_ids` and coordinates local to `index_origin`.
pub(crate) struct SparseAmr {
    pub(crate) component_ids: Vec<usize>,
    pub(crate) levels: Vec<SparseAmrLevel>,
}

pub(crate) struct SparseAmrLevel {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) index_type: IVec3,
    pub(crate) physical_origin: DVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) components: Vec<SparseGrid3<f32>>,
    pub(crate) valid_regions: Vec<Aabb3u>,
}

impl SparseAmr {
    pub(crate) fn load(plot_file: &PlotFile, component_ids: &[usize]) -> Result<Self> {
        for &component in component_ids {
            ensure!(
                component < plot_file.variables().len(),
                "component index {component} is out of range"
            );
        }

        let reader = DataReader::new(plot_file);
        let levels = plot_file
            .levels()
            .map(|level| {
                load_level(&reader, level, plot_file.header().domain.min, component_ids)
                    .with_context(|| format!("loading sparse AMR level {}", level.index()))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            component_ids: component_ids.to_vec(),
            levels,
        })
    }
}

fn load_level(
    reader: &DataReader<'_>,
    level: Level<'_>,
    physical_origin: DVec3,
    component_ids: &[usize],
) -> Result<SparseAmrLevel> {
    let domain = level.index_domain();
    let bounds = index_extent(domain)?;
    let mut components = component_ids
        .iter()
        .map(|_| SparseGrid3::with_chunk_capacity(bounds, level.patch_count()))
        .collect::<Vec<_>>();
    let mut valid_regions = Vec::with_capacity(level.patch_count());

    for patch in level.patches()? {
        let patch_box = patch.index_box();
        ensure!(
            patch_box.index_type == domain.index_type,
            "patch {} index type differs from its level domain",
            patch.index()
        );

        let local_aabb = local_aabb(patch_box, domain)?;
        for (&component, destination) in component_ids.iter().zip(&mut components) {
            load_patch_component(reader, patch, component, local_aabb, destination)?;
        }
        valid_regions.push(local_aabb);
    }

    Ok(SparseAmrLevel {
        level_index: level.index(),
        index_origin: domain.min,
        index_type: domain.index_type,
        physical_origin,
        cell_size: level.cell_size(),
        components,
        valid_regions,
    })
}

fn load_patch_component(
    reader: &DataReader<'_>,
    patch: Patch<'_>,
    component: usize,
    local_aabb: Aabb3u,
    destination: &mut SparseGrid3<f32>,
) -> Result<()> {
    let patch_box = patch.index_box();
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
                        "valid index {index:?} is absent from patch {} component {component}",
                        patch.index()
                    )
                })?;
                values.push(value as f32);
            }
        }
    }

    destination
        .write_aabb_values(local_aabb, &values)
        .map_err(|error| anyhow::anyhow!("writing patch component {component}: {error:?}"))
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

pub(crate) fn level_translation(
    coarse_origin: IVec3,
    fine_origin: IVec3,
    scale: u32,
) -> Result<I64Vec3> {
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
