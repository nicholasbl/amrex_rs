# amrex_rs

`amrex_rs` reads AMReX plotfiles and converts one plotfile timestep into the
source-independent `multires_rs::MsgDataset` model. MSG archive I/O,
isosurfacing, slicing, relaxation, and decimation live in the sibling
`multires_rs` crate.

The crate provides:

- parsing of AMReX `Header` and per-level `Cell_H` metadata;
- memory-mapped FAB component access;
- conversion of selected AMReX components into point-associated MSG fields;
- the `amrex-to-msg` command-line converter.

AMReX cell-center samples are preserved without interpolation and represented
as points using the MSG dataset's half-cell sample offset.

## Reading plotfiles

```rust
use amrex_rs::PlotFile;

let plotfile = PlotFile::open("plt00010")?;
println!("time = {}", plotfile.header().simulation_time);

for variable in plotfile.variables() {
    println!("{}: {}", variable.index, variable.name);
}

let density = plotfile.variable("density").unwrap();
let reader = plotfile.data_reader();
for level in plotfile.levels() {
    for patch in level.patches()? {
        let values = reader.variable(patch, density)?;
        println!("level {} patch {}: {} values", level.index(), patch.index(), values.len());
    }
}
# Ok::<(), anyhow::Error>(())
```

## Converting to MSG

```rust
use amrex_rs::{AmrexToMsgOptions, PlotFile, plotfile_to_msg};
use multires_rs::{MsgWriteOptions, write_msg};

let plotfile = PlotFile::open("plt00010")?;
let (dataset, report) = plotfile_to_msg(&plotfile, &AmrexToMsgOptions::default())?;

let mut bytes = Vec::new();
write_msg(&dataset, &MsgWriteOptions::default(), &mut bytes)?;
println!("converted {} scalar samples", report.sample_count);
# Ok::<(), anyhow::Error>(())
```

The CLI accepts input/output paths directly:

```sh
cargo run --release --bin amrex-to-msg -- plt00010 plt00010.msg
```

Repeat `--variable NAME` to select fields, or provide a reusable TOML file. When
`--config` is used, input and output may still be supplied on the command line
as a pair of overrides.

```toml
input = "plt00010"
output = "plt00010.msg"
replace_invalid = false

[[variables]]
name = "density"
scalar_type = "f64"
codec = "auto:smallest"
error = { absolute = 0.0, relative_percent = 0.01 }

[[variables]]
name = "temperature"
scalar_type = "f32"
codec = "auto:lossless"
```

MSG fields expose physical `f32` or `f64` values. Quantization and affine
encoding are archive codecs, not logical scalar types. Source values must be
finite; `replace_invalid = true` explicitly converts invalid or
unrepresentable source values to zero and reports their counts.

See `../multires_rs/README.md` for the `.msg` layout, codec families, and the
`msg-probe`, `msg-transcode`, and `msg-isosurface` tools.
