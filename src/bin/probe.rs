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
    println!("size: {file_size} bytes");
    println!("simulation time: {}", view.simulation_time());
    println!("coordinate system: {:?}", view.coordinate_system());
    println!(
        "domain: {} to {}",
        format_vec3(domain_min),
        format_vec3(domain_max)
    );
    println!("levels: {}", view.level_count());

    let index_domains = view.index_domains().collect::<Vec<_>>();
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
