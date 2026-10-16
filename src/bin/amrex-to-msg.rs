use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use amrex_rs::{
    AmrexInvalidValuePolicy, AmrexMsgField, AmrexToMsgOptions, PlotFile, plotfile_to_msg,
};
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use multires_rs::{
    MsgChunkCodec, MsgChunkCodecSelection, MsgErrorTolerance, MsgFieldCodec, MsgScalarType,
    MsgWriteOptions, write_msg,
};
use serde::Deserialize;

fn main() -> Result<()> {
    let args = ResolvedArgs::resolve(Cli::parse())?;
    let plotfile = PlotFile::open(&args.input)
        .with_context(|| format!("opening AMReX plotfile {}", args.input.display()))?;
    let mut fields = Vec::with_capacity(args.variables.len());
    let mut codecs = Vec::new();
    for requested in args.variables {
        let variable = plotfile
            .variable(&requested.name)
            .with_context(|| format!("variable {:?} was not found", requested.name))?;
        ensure!(
            !fields
                .iter()
                .any(|field: &AmrexMsgField| field.component == variable.index),
            "variable {:?} was requested more than once",
            requested.name
        );
        fields.push(AmrexMsgField {
            component: variable.index,
            scalar_type: requested
                .scalar_type
                .unwrap_or(ConfigScalarType::F32)
                .into(),
        });
        if requested.codec.is_some() || requested.error.is_some() {
            let selection = requested
                .codec
                .as_deref()
                .map(parse_codec)
                .transpose()?
                .context("an error tolerance requires an explicit codec")?;
            codecs.push(MsgFieldCodec {
                field: requested.name,
                selection,
                error_tolerance: requested
                    .error
                    .map(ConfigErrorTolerance::resolve)
                    .transpose()?,
            });
        }
    }

    let (dataset, report) = plotfile_to_msg(
        &plotfile,
        &AmrexToMsgOptions {
            fields,
            invalid_value_policy: if args.replace_invalid {
                AmrexInvalidValuePolicy::ReplaceWithZero
            } else {
                AmrexInvalidValuePolicy::Reject
            },
        },
    )?;
    let file = File::create(&args.output)
        .with_context(|| format!("creating MSG archive {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    let archive_start = Instant::now();
    write_msg(&dataset, &MsgWriteOptions { codecs }, &mut writer)
        .with_context(|| format!("writing MSG archive {}", args.output.display()))?;
    let archive_time = archive_start.elapsed();
    let flush_start = Instant::now();
    writer.flush()?;
    let flush_time = flush_start.elapsed();

    eprintln!(
        "AMReX to MSG timings: conversion {:.6}s, archive {:.6}s, final flush {:.6}s",
        report.elapsed.as_secs_f64(),
        archive_time.as_secs_f64(),
        flush_time.as_secs_f64()
    );
    eprintln!(
        "AMReX to MSG work: fields {}, levels {}, patches {}, samples {}",
        dataset.fields().len(),
        dataset.levels().len(),
        report.patch_count,
        report.sample_count
    );
    for (field, replacements) in dataset.fields().iter().zip(report.replaced_values) {
        if replacements != 0 {
            eprintln!(
                "warning: replaced {replacements} invalid values with zero in {:?}",
                field.name()
            );
        }
    }
    Ok(())
}

#[derive(Debug, Parser)]
#[command(
    about = "Convert one AMReX plotfile timestep into an MSG archive",
    after_help = "With --config, positional paths override both paths from the TOML file."
)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(short, long, value_name = "NAME", action = clap::ArgAction::Append, conflicts_with = "config")]
    variable: Vec<String>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
}

struct ResolvedArgs {
    input: PathBuf,
    output: PathBuf,
    variables: Vec<VariableConfig>,
    replace_invalid: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    #[serde(default)]
    variables: Vec<VariableConfig>,
    #[serde(default)]
    replace_invalid: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableConfig {
    name: String,
    scalar_type: Option<ConfigScalarType>,
    codec: Option<String>,
    error: Option<ConfigErrorTolerance>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigErrorTolerance {
    absolute: Option<f64>,
    relative_percent: Option<f64>,
}

impl ConfigErrorTolerance {
    fn resolve(self) -> Result<MsgErrorTolerance> {
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
        Ok(MsgErrorTolerance {
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
}

impl From<ConfigScalarType> for MsgScalarType {
    fn from(value: ConfigScalarType) -> Self {
        match value {
            ConfigScalarType::F32 => Self::F32,
            ConfigScalarType::F64 => Self::F64,
        }
    }
}

fn parse_codec(value: &str) -> Result<MsgChunkCodecSelection> {
    Ok(match value {
        "auto:lossless" => MsgChunkCodecSelection::AutoLossless,
        "auto:smallest" => MsgChunkCodecSelection::AutoSmallest,
        "uniform:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::UniformF64),
        "dense:f32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF32),
        "dense:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF64),
        "quantized:u8" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU8),
        "quantized:u16" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU16),
        "quantized:u32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU32),
        "affine:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineF64),
        "affine-residual:u8" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU8),
        "affine-residual:u16" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU16),
        "affine-residual:u32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU32),
        _ => bail!("unsupported MSG chunk codec {value:?}; expected family:value"),
    })
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
                _ => bail!("config path overrides require both INPUT and OUTPUT"),
            };
            return Ok(Self {
                input,
                output,
                variables: config.variables,
                replace_invalid: config.replace_invalid,
            });
        }

        let input = cli.input.context("missing INPUT plotfile")?;
        let output = cli.output.context("missing OUTPUT .msg path")?;
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
                    error: None,
                })
                .collect(),
            replace_invalid: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_msg_codec_names() {
        assert_eq!(
            parse_codec("quantized:u16").unwrap(),
            MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU16)
        );
        assert!(parse_codec("dense:u32").is_err());
    }
}
