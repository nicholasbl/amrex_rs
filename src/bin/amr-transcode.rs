use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use amrex_rs::{
    CompactChunkCodec, CompactChunkCodecSelection, CompactComponentCodec, CompactErrorTolerance,
    TranscodeCompactOptions, transcode_compact, view_compact,
};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

fn main() -> Result<()> {
    let args = Args::parse()?;
    reject_same_file(&args.input, &args.output)?;

    let input_file = File::open(&args.input)
        .with_context(|| format!("opening compact archive {}", args.input.display()))?;
    let input_size = input_file
        .metadata()
        .with_context(|| format!("reading metadata for {}", args.input.display()))?
        .len();
    // SAFETY: The read-only mapping remains alive and unchanged during transcoding.
    let bytes = unsafe {
        memmap2::Mmap::map(&input_file)
            .with_context(|| format!("memory mapping {}", args.input.display()))?
    };
    let view = view_compact(&bytes)
        .with_context(|| format!("reading compact archive {}", args.input.display()))?;
    let codecs = build_codec_policies(&view, &args.config)?;

    let output_file = File::create(&args.output)
        .with_context(|| format!("creating compact archive {}", args.output.display()))?;
    let mut writer = BufWriter::new(output_file);
    let timings = transcode_compact(&bytes, TranscodeCompactOptions { codecs }, &mut writer)
        .with_context(|| {
            format!(
                "transcoding {} to {}",
                args.input.display(),
                args.output.display()
            )
        })?;
    writer
        .flush()
        .with_context(|| format!("flushing compact archive {}", args.output.display()))?;
    let output_size = fs::metadata(&args.output)
        .with_context(|| format!("reading metadata for {}", args.output.display()))?
        .len();

    eprintln!(
        "timings: decode {}, archive {}, serialization {}",
        timings.decode_time.as_secs_f32(),
        timings.archive_time.as_secs_f32(),
        timings.serialization_time.as_secs_f32()
    );
    if input_size > 0 {
        eprintln!(
            "size: {} -> {} bytes ({:.2}% of source, {:.2}x source/output)",
            input_size,
            output_size,
            output_size as f64 / input_size as f64 * 100.0,
            input_size as f64 / output_size.max(1) as f64
        );
    }
    Ok(())
}

fn reject_same_file(input: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        ensure!(
            fs::canonicalize(input)? != fs::canonicalize(output)?,
            "input and output must be different files"
        );
    }
    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    config: Config,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    codec: Option<String>,
    error: Option<ConfigErrorTolerance>,
    #[serde(default)]
    variables: Vec<VariableConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableConfig {
    name: String,
    codec: String,
    error: Option<ConfigErrorTolerance>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigErrorTolerance {
    absolute: Option<f64>,
    range_percent: Option<f64>,
}

impl ConfigErrorTolerance {
    fn resolve(self) -> Result<CompactErrorTolerance> {
        match (self.absolute, self.range_percent) {
            (Some(value), None) => Ok(CompactErrorTolerance::Absolute(value)),
            (None, Some(percent)) => {
                ensure!(
                    percent.is_finite() && (0.0..=100.0).contains(&percent),
                    "range_percent must be finite and between 0 and 100"
                );
                Ok(CompactErrorTolerance::RangeFraction(percent / 100.0))
            }
            (None, None) => bail!("error must specify absolute or range_percent"),
            (Some(_), Some(_)) => bail!("error cannot specify both absolute and range_percent"),
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        const USAGE: &str = "usage:\n  amr-transcode <input_archive> <output_archive>\n  amr-transcode --config <config.toml> [<input_archive> <output_archive>]\nWithout a config, all stored quantities are rewritten losslessly using their native dense scalar codec.";
        let args = env::args().skip(1).collect::<Vec<_>>();
        if args.iter().any(|arg| arg == "-h" || arg == "--help") {
            println!("{USAGE}");
            std::process::exit(0);
        }

        if args.first().is_some_and(|arg| arg == "--config") {
            ensure!(args.len() == 2 || args.len() == 4, "{USAGE}");
            let config_path = PathBuf::from(&args[1]);
            let source = fs::read_to_string(&config_path)
                .with_context(|| format!("reading config {}", config_path.display()))?;
            let mut config: Config = toml::from_str(&source)
                .with_context(|| format!("parsing config {}", config_path.display()))?;
            let (input, output) = if args.len() == 4 {
                (PathBuf::from(&args[2]), PathBuf::from(&args[3]))
            } else {
                (
                    config
                        .input
                        .take()
                        .context("config has no input; provide one on the command line")?,
                    config
                        .output
                        .take()
                        .context("config has no output; provide one on the command line")?,
                )
            };
            return Ok(Self {
                input,
                output,
                config,
            });
        }

        ensure!(args.len() == 2, "{USAGE}");
        Ok(Self {
            input: PathBuf::from(&args[0]),
            output: PathBuf::from(&args[1]),
            config: Config::default(),
        })
    }
}

fn build_codec_policies(
    view: &amrex_rs::CompactArchiveView<'_>,
    config: &Config,
) -> Result<Vec<CompactComponentCodec>> {
    let stored_ids = view.component_ids().collect::<Vec<_>>();
    let variables = view.variables().collect::<Vec<_>>();
    let default = match (&config.codec, config.error) {
        (None, None) => None,
        (None, Some(_)) => bail!("a top-level error tolerance requires a top-level codec"),
        (Some(codec), error) => Some((parse_codec(codec)?, resolve_error(error)?)),
    };

    let mut policies = stored_ids
        .iter()
        .map(|&component_id| {
            default.map(|(selection, error_tolerance)| CompactComponentCodec {
                component_id: component_id as u32,
                selection,
                error_tolerance,
            })
        })
        .collect::<Vec<_>>();

    for (position, variable) in config.variables.iter().enumerate() {
        ensure!(
            !config.variables[..position]
                .iter()
                .any(|earlier| earlier.name == variable.name),
            "variable {:?} is configured more than once",
            variable.name
        );
        let component_id = variables
            .iter()
            .find_map(|&(name, id)| (name == variable.name).then_some(id))
            .with_context(|| format!("variable {:?} was not found", variable.name))?;
        let slot = stored_ids
            .iter()
            .position(|&stored| stored == component_id)
            .with_context(|| {
                format!("variable {:?} is not stored in the archive", variable.name)
            })?;
        policies[slot] = Some(CompactComponentCodec {
            component_id: component_id as u32,
            selection: parse_codec(&variable.codec)?,
            error_tolerance: resolve_error(variable.error)?,
        });
    }

    Ok(policies.into_iter().flatten().collect())
}

fn resolve_error(error: Option<ConfigErrorTolerance>) -> Result<Option<CompactErrorTolerance>> {
    error.map(ConfigErrorTolerance::resolve).transpose()
}

fn parse_codec(value: &str) -> Result<CompactChunkCodecSelection> {
    let selection = match value {
        "auto:lossless" => CompactChunkCodecSelection::AutoLossless,
        "auto:smallest" => CompactChunkCodecSelection::AutoSmallest,
        "uniform:f64" => CompactChunkCodecSelection::Forced(CompactChunkCodec::UniformF64),
        "dense:f32" => CompactChunkCodecSelection::Forced(CompactChunkCodec::DenseF32),
        "dense:f64" => CompactChunkCodecSelection::Forced(CompactChunkCodec::DenseF64),
        "dense:u32" => CompactChunkCodecSelection::Forced(CompactChunkCodec::DenseU32),
        "quantized:u8" => CompactChunkCodecSelection::Forced(CompactChunkCodec::QuantizedU8),
        "quantized:u16" => CompactChunkCodecSelection::Forced(CompactChunkCodec::QuantizedU16),
        "quantized:u32" => CompactChunkCodecSelection::Forced(CompactChunkCodec::QuantizedU32),
        "affine:f64" => CompactChunkCodecSelection::Forced(CompactChunkCodec::AffineF64),
        "affine-residual:u8" => {
            CompactChunkCodecSelection::Forced(CompactChunkCodec::AffineResidualU8)
        }
        "affine-residual:u16" => {
            CompactChunkCodecSelection::Forced(CompactChunkCodec::AffineResidualU16)
        }
        "affine-residual:u32" => {
            CompactChunkCodecSelection::Forced(CompactChunkCodec::AffineResidualU32)
        }
        _ => bail!("unsupported chunk codec {value:?}; expected family:value"),
    };
    Ok(selection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_transcode_config() -> Result<()> {
        let config: Config = toml::from_str(
            r#"
                input = "dense.compact"
                output = "lossy.compact"
                codec = "auto:smallest"
                error = { range_percent = 0.01 }

                [[variables]]
                name = "velocity_x"
                codec = "quantized:u16"
                error = { absolute = 1.0 }
            "#,
        )?;
        assert_eq!(config.variables.len(), 1);
        assert_eq!(
            config.error.unwrap().resolve()?,
            CompactErrorTolerance::RangeFraction(0.0001)
        );
        assert_eq!(
            config.variables[0].error.unwrap().resolve()?,
            CompactErrorTolerance::Absolute(1.0)
        );
        Ok(())
    }
}
