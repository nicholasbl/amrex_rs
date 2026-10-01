use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::DVec3;
use glam::IVec3;

use crate::PlotFile;
use crate::compact::CompactPlot;
use crate::sparse_amr::level_translation;
use crate::utility::SparseGrid3;

#[cfg(test)]
pub(super) use crate::compact::build_active_dual_cubes;

/// Cell-centered samples and eligible dual-lattice cube anchors for one AMR level.
///
/// Sparse samples for one AMR level. Samples covered by a finer level remain
/// available; only `active_cubes` is masked.
pub(crate) struct DualGridLevel<'a> {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) physical_origin: DVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) samples: &'a SparseGrid3<f32>,
    pub(crate) sampled_quantities: Vec<&'a SparseGrid3<f32>>,
    pub(crate) active_cubes: SparseGrid3<()>,
}

/// Build the isosurface-specific dual lattice from reusable sparse AMR data.
///
/// Valid fine-level sample coverage masks eligible cube anchors on the
/// immediately coarser level. Fine and coarse meshes are not stitched here.
pub(super) fn load_compact_for_isosurface(
    plot_file: &PlotFile,
    component: usize,
    sampled_components: &[usize],
) -> Result<CompactPlot> {
    ensure!(
        sampled_components.len() <= 2,
        "at most two sampled quantities are supported"
    );

    let component_ids = std::iter::once(component)
        .chain(sampled_components.iter().copied())
        .collect::<Vec<_>>();
    let compact_plot = CompactPlot::load(plot_file, &component_ids)?;
    ensure!(
        compact_plot.component_ids == component_ids,
        "compact component order changed while loading"
    );

    Ok(compact_plot)
}

pub(super) fn dual_grid_levels_from_compact<'a>(
    compact_plot: &'a CompactPlot,
    component: usize,
    sampled_components: &[usize],
    level_range: Option<&RangeInclusive<usize>>,
) -> Result<Vec<DualGridLevel<'a>>> {
    ensure!(
        sampled_components.len() <= 2,
        "at most two sampled quantities are supported"
    );

    let surface_slot = compact_plot
        .component_slot(component)
        .with_context(|| format!("compact plot does not contain component {component}"))?;
    let sampled_slots = sampled_components
        .iter()
        .map(|&component| {
            compact_plot
                .component_slot(component)
                .with_context(|| format!("compact plot does not contain component {component}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut levels: Vec<DualGridLevel<'a>> = Vec::with_capacity(compact_plot.levels.len());
    for compact_level in &compact_plot.levels {
        if !level_in_range(compact_level.level_index, level_range) {
            continue;
        }

        let samples = compact_level
            .components
            .get(surface_slot)
            .context("surface component slot is absent from compact level")?;

        if let Some(coarse) = levels.last_mut() {
            let ratio = compact_plot
                .refinement_ratios
                .get(compact_level.level_index - 1)
                .context("refinement ratio is absent for compact level")?;

            let ratio = u32::try_from(*ratio).context("refinement ratio does not fit in u32")?;

            ensure!(ratio > 0, "refinement ratio must be nonzero");

            let translation =
                level_translation(coarse.index_origin, compact_level.index_origin, ratio)?;

            coarse
                .active_cubes
                .mask_out_by_presence_scaled(samples, ratio, translation);
        }

        levels.push(DualGridLevel {
            level_index: compact_level.level_index,
            index_origin: compact_level.index_origin,
            physical_origin: compact_level.physical_origin,
            cell_size: compact_level.cell_size,
            samples,
            sampled_quantities: sampled_slots
                .iter()
                .map(|&slot| {
                    compact_level
                        .components
                        .get(slot)
                        .context("sample component slot is absent from compact level")
                })
                .collect::<Result<Vec<_>>>()?,
            active_cubes: compact_level.eligible_cubes.clone(),
        });
    }

    Ok(levels)
}

fn level_in_range(level_index: usize, level_range: Option<&RangeInclusive<usize>>) -> bool {
    level_range.is_none_or(|range| range.contains(&level_index))
}
