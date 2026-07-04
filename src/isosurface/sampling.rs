use anyhow::{Context, Result, ensure};
use glam::{U16Vec2, UVec3};

use crate::utility::SparseGrid3;

use super::SampleRange;
use super::dual_grid::DualGridLevel;

/// Interpolate auxiliary quantities along the same edge and with the same
/// parameter used to place an isosurface vertex.
pub(super) fn sampled_values_on_edge(
    level: &DualGridLevel,
    start: UVec3,
    end: UVec3,
    t: f32,
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    encode_sampled_values(level, ranges, |grid| {
        let start_value = grid
            .get(start)
            .with_context(|| format!("sample value is absent at edge start {start:?}"))?;
        let end_value = grid
            .get(end)
            .with_context(|| format!("sample value is absent at edge end {end:?}"))?;
        Ok((end_value - start_value).mul_add(t, start_value))
    })
}

/// Read auxiliary quantities at a lattice point used by an RMT-snapped vertex.
pub(super) fn sampled_values_at(
    level: &DualGridLevel,
    position: UVec3,
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    encode_sampled_values(level, ranges, |grid| {
        grid.get(position)
            .with_context(|| format!("sample value is absent at snapped vertex {position:?}"))
    })
}

/// Blend auxiliary quantities using the weights of an MC33 interior vertex.
pub(super) fn sampled_values_weighted(
    level: &DualGridLevel,
    positions: &[UVec3; 8],
    weights: &[f64; 8],
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    let weight_sum = weights.iter().sum::<f64>();
    ensure!(weight_sum > 0.0, "MC33 interior vertex has zero weight");
    encode_sampled_values(level, ranges, |grid| {
        let mut value = 0.0;
        for (&position, &weight) in positions.iter().zip(weights) {
            let sample = grid.get(position).with_context(|| {
                format!("sample value is absent at MC33 interior vertex corner {position:?}")
            })?;
            value += f64::from(sample) * weight;
        }
        Ok((value / weight_sum) as f32)
    })
}

fn encode_sampled_values<F>(
    level: &DualGridLevel,
    ranges: &[SampleRange],
    mut value: F,
) -> Result<U16Vec2>
where
    F: FnMut(&SparseGrid3<f32>) -> Result<f32>,
{
    ensure!(
        level.sampled_quantities.len() == ranges.len(),
        "sample grid and range counts differ"
    );
    ensure!(ranges.len() <= 2, "at most two UV axes are supported");

    let mut encoded = [0_u16; 2];
    for (axis, (grid, range)) in level.sampled_quantities.iter().zip(ranges).enumerate() {
        encoded[axis] = encode_unorm16(value(grid)?, *range);
    }
    Ok(U16Vec2::new(encoded[0], encoded[1]))
}

fn encode_unorm16(value: f32, range: SampleRange) -> u16 {
    let normalized = ((f64::from(value) - range.min) / (range.max - range.min)).clamp(0.0, 1.0);
    (normalized * f64::from(u16::MAX)).round() as u16
}
