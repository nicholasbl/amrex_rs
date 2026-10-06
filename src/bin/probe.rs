use std::{env, fs::File, path::PathBuf};

use amrex_rs::{CompactComponentStats, CompactScalarEncoding, read_compact, view_compact};
use anyhow::{Context, Result, ensure};

fn usage(program: &str) -> String {
    format!("Usage: {program} <compact-archive>")
}

fn parse_args() -> Result<Option<PathBuf>> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "probe".to_string());
    let values = args.collect::<Vec<_>>();

    if values
        .iter()
        .any(|value| value == "-h" || value == "--help")
    {
        println!("{}", usage(&program));
        return Ok(None);
    }

    ensure!(values.len() == 1, "{}", usage(&program));
    Ok(Some(PathBuf::from(&values[0])))
}

fn format_vec3<T: std::fmt::Display>(value: [T; 3]) -> String {
    format!("({}, {}, {})", value[0], value[1], value[2])
}

fn format_range(stats: CompactComponentStats) -> String {
    match (stats.min, stats.max) {
        (Some(min), Some(max)) => format!("min={min:<14.7e} max={max:<14.7e}"),
        _ => "min=n/a            max=n/a".to_string(),
    }
}

fn format_bytes(bytes: u128) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {} ({bytes} bytes)", UNITS[unit])
    }
}

fn scalar_width(encoding: CompactScalarEncoding) -> u128 {
    match encoding {
        CompactScalarEncoding::F32 | CompactScalarEncoding::UNorm32 { .. } => 4,
        CompactScalarEncoding::F64 => 8,
    }
}

fn dense_scalar_payload_size(
    index_domains: &[([i32; 3], [i32; 3], [i32; 3])],
    encodings: &[CompactScalarEncoding],
) -> Result<u128> {
    let bytes_per_cell = encodings.iter().copied().map(scalar_width).sum::<u128>();
    let mut total = 0_u128;

    for (level, &(min, max, _)) in index_domains.iter().enumerate() {
        let mut cell_count = 1_u128;
        for axis in 0..3 {
            let extent = i64::from(max[axis]) - i64::from(min[axis]) + 1;
            ensure!(extent > 0, "level {level} has an empty index domain");
            cell_count = cell_count
                .checked_mul(extent as u128)
                .context("dense cell count overflow")?;
        }
        total = total
            .checked_add(
                cell_count
                    .checked_mul(bytes_per_cell)
                    .context("dense level payload size overflow")?,
            )
            .context("dense payload size overflow")?;
    }

    Ok(total)
}

fn main() -> Result<()> {
    let Some(path) = parse_args()? else {
        return Ok(());
    };

    let file =
        File::open(&path).with_context(|| format!("opening compact archive {}", path.display()))?;
    let file_size = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len();
    // SAFETY: The read-only mapping remains alive and unchanged while it is used.
    let bytes = unsafe { memmap2::Mmap::map(&file).context("memory mapping compact archive")? };
    let view = view_compact(&bytes)
        .with_context(|| format!("reading compact archive metadata from {}", path.display()))?;

    let (domain_min, domain_max) = view.domain();
    println!("archive: {}", path.display());
    println!("size: {}", format_bytes(u128::from(file_size)));
    println!("simulation time: {}", view.simulation_time());
    println!("coordinate system: {:?}", view.coordinate_system());
    println!(
        "domain: {} to {}",
        format_vec3(domain_min),
        format_vec3(domain_max)
    );
    println!("levels: {}", view.level_count());

    let index_domains = view.index_domains().collect::<Vec<_>>();
    let component_encodings = view
        .component_encodings()
        .map(|(_, encoding)| encoding)
        .collect::<Vec<_>>();
    let dense_size = dense_scalar_payload_size(&index_domains, &component_encodings)?;
    println!(
        "dense scalar payload estimate: {}",
        format_bytes(dense_size)
    );
    if dense_size > 0 {
        let archive_fraction = file_size as f64 / dense_size as f64;
        let dense_to_archive = dense_size as f64 / file_size as f64;
        println!(
            "archive / dense payload: {:.2}% (dense/archive: {dense_to_archive:.2}x)",
            archive_fraction * 100.0
        );
    }
    let cell_sizes = view.cell_sizes().collect::<Vec<_>>();
    let level_steps = view.level_steps().collect::<Vec<_>>();
    for level in 0..view.level_count() {
        let (min, max, index_type) = index_domains
            .get(level)
            .copied()
            .with_context(|| format!("archive has no index domain for level {level}"))?;
        let cell_size = cell_sizes
            .get(level)
            .copied()
            .with_context(|| format!("archive has no cell size for level {level}"))?;
        let step = level_steps
            .get(level)
            .with_context(|| format!("archive has no step counter for level {level}"))?;
        println!(
            "  level {level}: step={step}, cells={} to {}, index_type={}, cell_size={}",
            format_vec3(min),
            format_vec3(max),
            format_vec3(index_type),
            format_vec3(cell_size)
        );
    }

    let compact = read_compact(&bytes)
        .with_context(|| format!("loading compact archive {}", path.display()))?;
    ensure!(
        compact.level_count() == view.level_count(),
        "archive metadata and compact data disagree on level count"
    );

    println!(
        "stored variables: {} of {}",
        compact.component_ids().len(),
        compact.variable_count()
    );
    for &component_id in compact.component_ids() {
        let variable = compact
            .variables()
            .iter()
            .find(|variable| variable.index == component_id)
            .with_context(|| format!("no variable metadata for component {component_id}"))?;
        let stats = compact
            .component_stats(component_id)
            .with_context(|| format!("component {component_id} is not stored"))?;
        let nan_suffix = if stats.nan_count == 0 {
            String::new()
        } else {
            format!(", NaNs={}", stats.nan_count)
        };
        let normalization_suffix = compact
            .component_normalization(component_id)
            .map(|(min, max)| format!(", normalized from [{min:.7e}, {max:.7e}]"))
            .unwrap_or_default();
        let encoding = match compact.component_encoding(component_id) {
            Some(CompactScalarEncoding::F32) => "f32",
            Some(CompactScalarEncoding::F64) => "f64",
            Some(CompactScalarEncoding::UNorm32 { .. }) => "unorm32",
            None => "unknown",
        };
        println!(
            "  [{component_id:>3}] {:<24} {} values={} encoding={encoding}{nan_suffix}{normalization_suffix}",
            variable.name,
            format_range(stats),
            stats.value_count
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_estimate_uses_inclusive_domains_and_encoding_widths() -> Result<()> {
        let domains = [([0, 0, 0], [1, 2, 3], [0, 0, 0])];
        let encodings = [
            CompactScalarEncoding::F32,
            CompactScalarEncoding::F64,
            CompactScalarEncoding::UNorm32 { min: 0.0, max: 1.0 },
        ];

        // 2 * 3 * 4 cells, with 4 + 8 + 4 bytes per cell.
        assert_eq!(dense_scalar_payload_size(&domains, &encodings)?, 384);
        Ok(())
    }

    #[test]
    fn byte_format_is_human_readable() {
        assert_eq!(format_bytes(1000), "1000 B");
        assert_eq!(format_bytes(1536), "1.50 KiB (1536 bytes)");
    }
}
