mod dual_grid;
mod mc33;
mod rmt;
mod sampling;

use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::{U16Vec2, UVec3, Vec3};

use crate::PlotFile;

use dual_grid::load_dual_grid_levels;

#[derive(Debug, Default)]
pub struct Surface {
    pub id: u32,
    pub value: f64,
}

#[derive(Debug)]
pub struct Sample {
    pub id: u32,
    pub range: RangeInclusive<f64>,
}

#[derive(Debug, Default)]
pub struct IsosurfaceOptions {
    pub surface: Surface,
    pub sampled_quantities: Vec<Sample>,
    pub method: IsosurfaceMethod,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum IsosurfaceMethod {
    #[default]
    Mc33,
    Rmt {
        /// Fraction of an edge near either endpoint that is snapped to that endpoint.
        regularization: f32,
    },
}

#[derive(Debug)]
pub struct Mesh3D {
    pub positions: Vec<Vertex3D>,
    pub faces: Vec<UVec3>,
}

#[derive(Debug)]
pub struct Vertex3D {
    pub position: Vec3,
    pub sampled_values: U16Vec2,
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    min: f64,
    max: f64,
}

#[derive(Debug, Clone, Copy)]
struct SampleSpec {
    component: usize,
    range: SampleRange,
}

fn validate_sample_specs(plot_file: &PlotFile, samples: &[Sample]) -> Result<Vec<SampleSpec>> {
    ensure!(
        samples.len() <= 2,
        "at most two sampled quantities are supported"
    );

    samples
        .iter()
        .map(|sample| {
            let component =
                usize::try_from(sample.id).context("sample component id is too large")?;
            ensure!(
                component < plot_file.variables().len(),
                "sample component index {component} is out of range"
            );

            let min = *sample.range.start();
            let max = *sample.range.end();
            ensure!(
                min.is_finite() && max.is_finite(),
                "sample range bounds must be finite"
            );
            ensure!(max > min, "sample range maximum must exceed its minimum");

            Ok(SampleSpec {
                component,
                range: SampleRange { min, max },
            })
        })
        .collect()
}

pub fn isosurface(plot_file: &PlotFile, options: IsosurfaceOptions) -> Result<Mesh3D> {
    let component = usize::try_from(options.surface.id).context("surface id is too large")?;
    ensure!(options.surface.value.is_finite(), "isovalue must be finite");
    if let IsosurfaceMethod::Rmt { regularization } = options.method {
        ensure!(
            regularization.is_finite() && (0.0..0.5).contains(&regularization),
            "regularization must be in [0, 0.5)"
        );
    }
    let isovalue = options.surface.value as f32;
    ensure!(isovalue.is_finite(), "isovalue does not fit in f32");

    let sample_specs = validate_sample_specs(plot_file, &options.sampled_quantities)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let levels = load_dual_grid_levels(plot_file, component, &sampled_components)?;
    let ranges = sample_specs
        .iter()
        .map(|sample| sample.range)
        .collect::<Vec<_>>();
    let mut mesh = Mesh3D {
        positions: Vec::new(),
        faces: Vec::new(),
    };

    for level in &levels {
        match options.method {
            IsosurfaceMethod::Mc33 => mc33::mesh_level(level, &ranges, isovalue, &mut mesh),
            IsosurfaceMethod::Rmt { regularization } => {
                rmt::mesh_level(level, &ranges, isovalue, regularization, &mut mesh)
            }
        }
        .with_context(|| format!("extracting level {}", level.level_index))?;
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests;
