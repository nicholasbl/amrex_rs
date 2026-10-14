# amrex_rs

Rust utilities for reading AMReX plotfiles and extracting MC33 isosurfaces from cell-centered data.

The crate currently focuses on:

- parsing AMReX plotfile `Header` and per-level `Cell_H` metadata
- memory-mapped reads of FAB component data
- compact sparse AMR archives for selected components
- MC33 isosurface extraction with optional sampled vertex quantities

Inspect a compact archive and report its metadata and value ranges:

```text
cargo run --release --bin amr-probe -- archive.compact
```

## Basic Usage

Open a plotfile and inspect its metadata:

```rust
use amrex_rs::PlotFile;

let plotfile = PlotFile::open("plt00010")?;
println!("time = {}", plotfile.header().simulation_time);

for variable in plotfile.variables() {
    println!("{}: {}", variable.index, variable.name);
}
```

Read component data patch by patch:

```rust
use amrex_rs::PlotFile;

let plotfile = PlotFile::open("plt00010")?;
let density = plotfile.variable("density").unwrap();
let reader = plotfile.data_reader();

for level in plotfile.levels() {
    for patch in level.patches()? {
        let values = reader.variable(patch, density)?;
        println!(
            "level {} patch {} has {} values",
            patch.level_index(),
            patch.index(),
            values.len()
        );
    }
}
```

Extract an isosurface:

```rust
use amrex_rs::{IsosurfaceOptions, PlotFile, Surface, isosurface};

let plotfile = PlotFile::open("plt00010")?;
let density = plotfile.variable("density").unwrap();
let (mesh, _) = isosurface(
    &plotfile,
    IsosurfaceOptions {
        surface: Surface {
            id: density.index as u32,
            value: 1.0,
        },
        sampled_quantities: Vec::new(),
        levels: None,
        flip_winding: false,
    },
)?;
println!(
    "{} vertices, {} triangles",
    mesh.positions.len(),
    mesh.indices.len()
);
```

## OBJ Example

The included binary writes an OBJ mesh:

```sh
cargo run --bin amr-isosurface -- \
  /path/to/plt00010 density 1.0 surface.obj
```

Up to two additional variables can be sampled onto `mesh.uv` and written as OBJ texture coordinates:

```sh
cargo run --bin amr-isosurface -- \
  /path/to/plt00010 density 1.0 surface.obj \
  --sample temperature 0.0 10.0 \
  --sample pressure 1.0 5.0
```

## Mesh Decimation

Clean up and decimate an extracted mesh with the meshoptimizer-backed convenience pipeline:

```rust
use amrex_rs::{
    DecimateOptions, DecimatePipelineOptions, DecimateTarget, decimate_mesh_pipeline,
};

let result = decimate_mesh_pipeline(
    &mut mesh,
    DecimatePipelineOptions {
        decimate: DecimateOptions {
            target: DecimateTarget::FaceRatio(0.25),
            max_error: 0.01,
            ..DecimateOptions::default()
        },
        ..DecimatePipelineOptions::default()
    },
)?;

println!(
    "{} -> {} faces",
    result.decimation.original_face_count,
    result.decimation.final_face_count
);
println!(
    "degenerates {:?}, pre-compact {:?}, decimate {:?}, total {:?}",
    result.timings.degenerate_removal,
    result.timings.pre_decimation_compaction,
    result.timings.decimation,
    result.timings.total,
);
println!(
    "meshoptimizer {:?}, groups {} ({} faces before final), validation {:?} + {:?}, post-compact {:?}",
    result.decimation.timings.simplification,
    result.decimation.group_count,
    result.decimation.intermediate_face_count,
    result.decimation.timings.input_validation,
    result.decimation.timings.output_validation,
    result.decimation.timings.compaction,
);
```

MC33 vertices shared by parallel extraction chunks are welded exactly from
their grid edge/cell keys while each level is merged. The decimation pipeline
therefore starts with degenerate-face removal and vertex compaction; it does not
perform approximate geometric welding. Use `decimate_mesh` directly when the
mesh is already clean. Error limits can be relative to mesh extent or absolute
in the physical units used by `Mesh3D::positions`.

Large meshes use a topology-aware parallel path by default. Meshlets seed
larger connected partitions; Rayon simplifies those partitions with locked
borders, then a final sparse global pass removes the temporary partition
boundaries. Configure the threshold and group sizes through
`DecimateParallelOptions`, or set `DecimateOptions::parallel` to `None` to force
the original serial path. Detailed timings are available in `DecimateTimings`
for meshlet construction, partitioning, grouped simplification, merging, and
the final global simplification.

## Vertex Relaxation

Apply optional Taubin smoothing directly to any cleaned `Mesh3D`:

```rust
use amrex_rs::{VertexRelaxOptions, relax_mesh_vertices};

let result = relax_mesh_vertices(&mut mesh, VertexRelaxOptions::default())?;
println!(
    "relaxed {} vertices ({} locked) in {:?}",
    result.relaxed_vertex_count,
    result.locked_vertex_count,
    result.timings.total,
);
```

Relaxation is deliberately separate from isosurface extraction and decimation,
so callers can choose when to apply it. The default performs three
shrinkage-resistant Taubin pass pairs, clamps each step relative to local edge
length, locks open and non-manifold boundaries, and leaves UV/scientific values
unchanged. Clean repeated-index faces before relaxing the mesh.

## Compact Archives

The `amr-compact` binary converts an AMReX plotfile directory into a single compact
archive. This is useful when repeatedly extracting surfaces from the same data: the
archive stores the selected components in the sparse layout used by the isosurface
code, so the original FAB files do not need to be read and compacted on every run.

The command takes the input plotfile and output archive. Repeat `--variable` to
select quantities explicitly:

```text
amr-compact [--variable <name>]... <input_plt> <output_archive>
```

Variable names are case-sensitive. To archive only `density` and `temperature`:

```sh
cargo run --release --bin amr-compact -- \
  --variable density --variable temperature \
  /path/to/plt00010 plt00010.compact
```

Omit `--variable` to include every variable in the plotfile:

```sh
cargo run --release --bin amr-compact -- \
  /path/to/plt00010 plt00010.compact
```

On completion, `amr-compact` reports the compact-load, archive-building, and
serialization timings to standard error in seconds.

The `amr-isosurface` binary accepts the resulting archive anywhere it accepts a
plotfile directory. The surface variable and every `--sample` variable must have
been included when the archive was created:

```sh
cargo run --release --bin amr-isosurface -- \
  plt00010.compact density 1.0 surface.obj \
  --sample temperature 0.0 10.0
```

The same functionality is available through the library API:

```rust
use amrex_rs::{CompactOptions, PlotFile, read_compact, write_compact};

let plotfile = PlotFile::open("plt00010")?;
let mut bytes = Vec::new();
write_compact(
    &plotfile,
    CompactOptions {
        component_ids: vec![0],
        ..CompactOptions::default()
    },
    &mut bytes,
)?;

let compact = read_compact(&bytes)?;
println!("original variable count = {}", compact.variable_count());
```

An empty `component_ids` list selects all variables. Otherwise, component IDs are
the zero-based indices exposed by `PlotFile::variables()`.

For per-variable storage, pass a reusable TOML configuration to `amr-compact`.
`scalar_type` controls the values reconstructed by the reader. Quantities
default to `f32`; `f64` retains genuine double precision, while `unorm32` maps
one global physical range to all 32 bits of an unsigned integer:

```toml
[[variables]]
name = "density"
scalar_type = "unorm32"
codec = "auto:smallest"
normalize = [1.0e20, 1.000000001e20]
error = { range_percent = 0.01 }

[[variables]]
name = "temperature"
scalar_type = "f64"
codec = "quantized:u16"
error = { absolute = 1.0e-4 }
```

`codec` controls the on-disk representation independently of `scalar_type`.
Every codec name uses `family:value` syntax. Automatic policies are
`auto:lossless` and `auto:smallest`. The latter selects the smallest supported
codec that meets the configured error tolerance. Concrete codecs force every non-uniform chunk
to use that codec: `dense:f32`, `dense:f64`, `dense:u32`, `uniform:f64`,
`quantized:u8`, `quantized:u16`, `quantized:u32`, `affine:f64`, and
`affine-residual:u8`, `affine-residual:u16`, or `affine-residual:u32`.
Structurally uniform sparse chunks always retain the exact `uniform:f64`
representation.

Lossy policies require an `error` table; exact native dense codecs and
`auto:lossless` reject it. `absolute` is measured after decoding in the
reconstructed scalar domain. `range_percent` is a percentage of the quantity's
global observed range in that same domain and is resolved once for all chunks.
For example, `range_percent = 0.01` means 0.01%, not 1%. A constant quantity
resolves to zero error. The reconstructed domain for `unorm32` is `[0, 1]`.
Omitting `codec` uses `auto:lossless`.

Chunk codecs have independent numeric ids and versions in the archive. Their
metadata and payloads are length-delimited, allowing future codecs to be added
without changing the container format. `amr-probe` reports the codec mix, payload
sizes, and maximum observed error bound without decoding scalar values.

```sh
cargo run --release --bin amr-compact -- \
  --config compact.toml /path/to/plt00010 plt00010.compact
```

This lets the same variable recipe be reused across a collection of plotfiles.
`input` and `output` may still be placed in the TOML and the shorter
`--config compact.toml` form used. When command-line paths are supplied, they
override the paths from the TOML.

For convenience, `normalize` without an explicit `scalar_type` implies
`unorm32`. Global normalization is performed in `f64` as
`round(clamp((value - min) / (max - min), 0, 1) * u32::MAX)`. The original
bounds are retained in archive metadata and shown by `amr-probe`. Isovalues for
normalized quantities must likewise be in the normalized `[0, 1]` range.
Omitting `variables` selects every quantity as `f32`.

By default, compaction replaces non-finite or non-representable values with
zero and prints aggregate warnings. Set `assume_finite = true` at the top level
of a trusted config to skip those checks. Input and output may be supplied in
the config or overridden on the command line.

To compare codec choices without rereading the original plotfile, transcode an
existing compact archive. With no configuration, every stored quantity is
rewritten losslessly using the dense codec native to its reconstructed scalar
type:

```sh
cargo run --release --bin amr-transcode -- \
  source.compact dense-reference.compact
```

Lossy policies can be applied globally with per-variable overrides:

```toml
input = "dense-reference.compact"
output = "test-0.01-percent.compact"
codec = "auto:smallest"
error = { range_percent = 0.01 }

[[variables]]
name = "velocity_x"
codec = "quantized:u16"
error = { absolute = 1.0 }
```

```sh
cargo run --release --bin amr-transcode -- --config transcode.toml
```

Command-line input and output paths may follow the config path to override its
paths. Transcoding preserves each quantity's reconstructed scalar type and all
AMR metadata. It operates on decoded values, so transcoding a lossy archive
cannot recover precision already discarded; use a lossless dense archive as
the source when comparing codecs.

## Notes

Isosurface extraction expects cell-centered data. Compact archives contain a
conservative min/max range for each sparse work chunk, including the positive
sample halo. Chunks that cannot contain the requested isovalue are rejected
before MC33 visits individual cubes. Scalar calculations remain in `f64`; final
mesh positions and UVs are stored as `f32` for graphics use.

The compact archive format is intended for use by this crate and may change before a stable 1.0 release.
The current offset-based format is version 6. Earlier archives must be
regenerated with the current `amr-compact`.
