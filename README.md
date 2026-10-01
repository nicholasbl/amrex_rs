# amrex_rs

Rust utilities for reading AMReX plotfiles and extracting MC33 isosurfaces from cell-centered data.

The crate currently focuses on:

- parsing AMReX plotfile `Header` and per-level `Cell_H` metadata
- memory-mapped reads of FAB component data
- compact sparse AMR archives for selected components
- MC33 isosurface extraction with optional sampled vertex quantities

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
cargo run --bin isosurface -- \
  /path/to/plt00010 density 1.0 surface.obj
```

Up to two additional variables can be sampled onto `mesh.uv` and written as OBJ texture coordinates:

```sh
cargo run --bin isosurface -- \
  /path/to/plt00010 density 1.0 surface.obj \
  --sample temperature 0.0 10.0 \
  --sample pressure 1.0 5.0
```

## Compact Archives

The `pltcompact` binary converts an AMReX plotfile directory into a single compact
archive. This is useful when repeatedly extracting surfaces from the same data: the
archive stores the selected components in the sparse layout used by the isosurface
code, so the original FAB files do not need to be read and compacted on every run.

The command takes zero or more variable names, followed by the input plotfile and
output archive:

```text
pltcompact [<variable> ...] <input_plt> <output_archive>
```

Variable names are case-sensitive. To archive only `density` and `temperature`:

```sh
cargo run --release --bin pltcompact -- \
  density temperature /path/to/plt00010 plt00010.compact
```

Omit the variable names to include every variable in the plotfile:

```sh
cargo run --release --bin pltcompact -- \
  /path/to/plt00010 plt00010.compact
```

The `isosurface` binary accepts the resulting archive anywhere it accepts a
plotfile directory. The surface variable and every `--sample` variable must have
been included when the archive was created:

```sh
cargo run --release --bin isosurface -- \
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
    },
    &mut bytes,
)?;

let compact = read_compact(&bytes)?;
println!("original variable count = {}", compact.variable_count());
```

An empty `component_ids` list selects all variables. Otherwise, component IDs are
the zero-based indices exposed by `PlotFile::variables()`.

## Notes

Isosurface extraction expects cell-centered data. The mesh path uses a sparse chunked grid internally, with cached accessors in the MC33 hot path to avoid repeated hash-map lookups inside the same chunk.

The compact archive format is intended for use by this crate and may change before a stable 1.0 release.
