use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use amrex_rs::{
    CompactChunkCodec, CompactChunkCodecSelection, CompactComponentCodec,
    CompactComponentScalarType, CompactErrorTolerance, CompactOptions, CompactScalarType,
    NonFinitePolicy, PlotFile, write_compact,
};
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::Deserialize;

fn main() -> Result<()> {
    let args = ResolvedArgs::resolve(Cli::parse())?;

    let plotfile = PlotFile::open(&args.input)
        .with_context(|| format!("opening plotfile {}", args.input.display()))?;

    let mut ids = Vec::with_capacity(args.variables.len());
    let mut scalar_types = Vec::new();
    let mut codecs = Vec::new();
    for requested in args.variables {
        let variable = plotfile
            .variable(&requested.name)
            .with_context(|| format!("variable {:?} was not found", requested.name))?;
        let component_id = u32::try_from(variable.index).context("component id exceeds u32")?;
        ensure!(
            !ids.contains(&component_id),
            "variable {:?} was requested more than once",
            requested.name
        );
        ids.push(component_id);
        let scalar_type = match (requested.scalar_type, requested.normalize) {
            (None | Some(ConfigScalarType::F32), None) => CompactScalarType::F32,
            (Some(ConfigScalarType::F64), None) => CompactScalarType::F64,
            (None | Some(ConfigScalarType::UNorm32), Some([min, max])) => {
                CompactScalarType::UNorm32 { min, max }
            }
            (Some(ConfigScalarType::UNorm32), None) => {
                bail!(
                    "variable {:?} uses unorm32 but has no normalize bounds",
                    requested.name
                )
            }
            (Some(scalar_type), Some(_)) => bail!(
                "variable {:?} has normalize bounds incompatible with {scalar_type:?}",
                requested.name
            ),
        };
        scalar_types.push(CompactComponentScalarType {
            component_id,
            scalar_type,
        });
        if requested.codec.is_some() || requested.error.is_some() {
            let selection = requested
                .codec
                .as_deref()
                .map(parse_codec)
                .transpose()?
                .context("an error tolerance requires an explicit codec")?;
            codecs.push(CompactComponentCodec {
                component_id,
                selection,
                error_tolerance: requested
                    .error
                    .map(ConfigErrorTolerance::resolve)
                    .transpose()?,
            });
        }
    }

    let file = File::create(&args.output)
        .with_context(|| format!("creating compact archive {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);

    let timings = write_compact(
        &plotfile,
        CompactOptions {
            component_ids: ids,
            scalar_types,
            codecs,
            non_finite_policy: if args.assume_finite {
                NonFinitePolicy::AssumeFinite
            } else {
                NonFinitePolicy::ReplaceWithZero
            },
        },
        &mut writer,
    )
    .with_context(|| format!("writing compact archive {}", args.output.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing compact archive {}", args.output.display()))?;

    eprintln!(
        "timings: compact {}, archive {}, serialization {}",
        timings.compact_time.as_secs_f32(),
        timings.archive_time.as_secs_f32(),
        timings.serialization_time.as_secs_f32()
    );
    for replacement in timings.non_finite_replacements {
        let name = plotfile
            .variables()
            .get(replacement.component_id as usize)
            .map_or("<unknown>", |variable| variable.name.as_str());
        eprintln!(
            "warning: replaced {} non-finite or unrepresentable values with zero in {name:?}",
            replacement.count
        );
    }

    Ok(())
}

#[derive(Debug, Parser)]
#[command(
    about = "Convert an AMReX plotfile into a compact AMR archive",
    after_help = "With --config, positional paths are optional overrides for paths in the TOML file."
)]
struct Cli {
    /// Reusable TOML compaction configuration.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Quantity to include. Repeat to select multiple quantities; omit for all.
    #[arg(short, long, value_name = "NAME", action = clap::ArgAction::Append, conflicts_with = "config")]
    variable: Vec<String>,
    /// Source AMReX plotfile directory.
    input: Option<PathBuf>,
    /// Destination compact archive.
    output: Option<PathBuf>,
}

struct ResolvedArgs {
    input: PathBuf,
    output: PathBuf,
    variables: Vec<VariableConfig>,
    assume_finite: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    #[serde(default)]
    variables: Vec<VariableConfig>,
    #[serde(default)]
    assume_finite: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableConfig {
    name: String,
    scalar_type: Option<ConfigScalarType>,
    codec: Option<String>,
    normalize: Option<[f64; 2]>,
    error: Option<ConfigErrorTolerance>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigErrorTolerance {
    absolute: Option<f64>,
    relative_percent: Option<f64>,
}

impl ConfigErrorTolerance {
    fn resolve(self) -> Result<CompactErrorTolerance> {
        ensure!(
            self.absolute.is_some() || self.relative_percent.is_some(),
            "error must specify absolute, relative_percent, or both"
        );
        let absolute = self.absolute.unwrap_or(0.0);
        let relative_percent = self.relative_percent.unwrap_or(0.0);
        ensure!(
            absolute.is_finite() && absolute >= 0.0,
            "absolute error must be finite and non-negative"
        );
        ensure!(
            relative_percent.is_finite() && (0.0..=100.0).contains(&relative_percent),
            "relative_percent must be finite and between 0 and 100"
        );
        Ok(CompactErrorTolerance {
            absolute,
            relative_percent,
        })
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConfigScalarType {
    F32,
    F64,
    UNorm32,
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

impl ResolvedArgs {
    fn resolve(cli: Cli) -> Result<Self> {
        if let Some(path) = cli.config {
            let source = fs::read_to_string(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let config: Config = toml::from_str(&source)
                .with_context(|| format!("parsing config {}", path.display()))?;
            let (input, output) = match (cli.input, cli.output) {
                (Some(input), Some(output)) => (input, output),
                (None, None) => (
                    config
                        .input
                        .context("config has no input; provide one on the command line")?,
                    config
                        .output
                        .context("config has no output; provide one on the command line")?,
                ),
                _ => {
                    bail!("--config path overrides require both INPUT_PLOTFILE and OUTPUT_ARCHIVE")
                }
            };
            return Ok(Self {
                input,
                output,
                variables: config.variables,
                assume_finite: config.assume_finite,
            });
        }

        let input = cli.input.context("missing INPUT_PLOTFILE")?;
        let output = cli.output.context("missing OUTPUT_ARCHIVE")?;

        Ok(Self {
            input,
            output,
            variables: cli
                .variable
                .into_iter()
                .map(|name| VariableConfig {
                    name,
                    scalar_type: None,
                    codec: None,
                    normalize: None,
                    error: None,
                })
                .collect(),
            assume_finite: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clap_parses_repeated_variable_options() -> Result<()> {
        let cli = Cli::try_parse_from([
            "amr-compact",
            "--variable",
            "density",
            "--variable",
            "temperature",
            "plt00010",
            "output.compact",
        ])?;
        let args = ResolvedArgs::resolve(cli)?;
        assert_eq!(args.input, PathBuf::from("plt00010"));
        assert_eq!(args.output, PathBuf::from("output.compact"));
        assert_eq!(
            args.variables
                .iter()
                .map(|variable| variable.name.as_str())
                .collect::<Vec<_>>(),
            ["density", "temperature"]
        );
        Ok(())
    }

    #[test]
    fn parses_toml_variable_normalization() -> Result<()> {
        let config: Config = toml::from_str(
            r#"
                input = "plt00010"
                output = "plt00010.compact"
                assume_finite = true

                [[variables]]
                name = "density"
                scalar_type = "unorm32"
                codec = "auto:smallest"
                normalize = [1.0e20, 1.000000001e20]
                error = { absolute = 1.0e10, relative_percent = 0.01 }

                [[variables]]
                name = "temperature"
                scalar_type = "f64"
                codec = "dense:f64"
            "#,
        )?;

        ensure!(config.variables.len() == 2, "unexpected variable count");
        ensure!(
            config.input == Some(PathBuf::from("plt00010")),
            "input path changed while parsing"
        );
        ensure!(
            config.output == Some(PathBuf::from("plt00010.compact")),
            "output path changed while parsing"
        );
        ensure!(
            config.variables[0].normalize == Some([1.0e20, 1.000000001e20]),
            "normalization bounds changed while parsing"
        );
        ensure!(
            config.variables[0].error.unwrap().resolve()?
                == CompactErrorTolerance {
                    absolute: 1.0e10,
                    relative_percent: 0.01,
                }
        );
        ensure!(
            parse_codec(config.variables[0].codec.as_deref().unwrap())?
                == CompactChunkCodecSelection::AutoSmallest
        );
        ensure!(
            config.variables[1].normalize.is_none(),
            "optional normalization unexpectedly present"
        );
        ensure!(config.assume_finite, "assume_finite changed while parsing");
        ensure!(
            matches!(
                config.variables[0].scalar_type,
                Some(ConfigScalarType::UNorm32)
            ) && matches!(config.variables[1].scalar_type, Some(ConfigScalarType::F64)),
            "scalar types changed while parsing"
        );
        Ok(())
    }

    #[test]
    fn config_paths_are_optional() -> Result<()> {
        let config: Config = toml::from_str(
            r#"
                [[variables]]
                name = "density"
                normalize = [0.0, 1.0]
            "#,
        )?;

        ensure!(config.input.is_none(), "unexpected config input");
        ensure!(config.output.is_none(), "unexpected config output");
        ensure!(config.variables.len() == 1, "unexpected variable count");
        Ok(())
    }

    #[test]
    fn parses_standard_codec_names() -> Result<()> {
        let cases = [
            ("auto:lossless", CompactChunkCodecSelection::AutoLossless),
            ("auto:smallest", CompactChunkCodecSelection::AutoSmallest),
            (
                "dense:f64",
                CompactChunkCodecSelection::Forced(CompactChunkCodec::DenseF64),
            ),
            (
                "quantized:u16",
                CompactChunkCodecSelection::Forced(CompactChunkCodec::QuantizedU16),
            ),
            (
                "affine-residual:u8",
                CompactChunkCodecSelection::Forced(CompactChunkCodec::AffineResidualU8),
            ),
        ];
        for (name, expected) in cases {
            assert_eq!(parse_codec(name)?, expected);
        }
        assert!(parse_codec("raw:f64").is_err());
        assert!(parse_codec("quantized:16").is_err());
        Ok(())
    }

    #[test]
    fn parses_absolute_and_relative_percent_tolerances() -> Result<()> {
        let absolute: ConfigErrorTolerance = toml::from_str("absolute = 2.5e12")?;
        assert_eq!(
            absolute.resolve()?,
            CompactErrorTolerance {
                absolute: 2.5e12,
                relative_percent: 0.0,
            }
        );

        let relative: ConfigErrorTolerance = toml::from_str("relative_percent = 0.01")?;
        assert_eq!(
            relative.resolve()?,
            CompactErrorTolerance {
                absolute: 0.0,
                relative_percent: 0.01,
            }
        );

        let both: ConfigErrorTolerance = toml::from_str("absolute = 1.0\nrelative_percent = 1.0")?;
        assert_eq!(
            both.resolve()?,
            CompactErrorTolerance {
                absolute: 1.0,
                relative_percent: 1.0,
            }
        );
        Ok(())
    }
}
