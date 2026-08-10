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
let mesh = isosurface(
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

Use compact archives when you want to load selected components once and reuse them:

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

## Notes

Isosurface extraction expects cell-centered data. The mesh path uses a sparse chunked grid internally, with cached accessors in the MC33 hot path to avoid repeated hash-map lookups inside the same chunk.

The compact archive format is intended for use by this crate and may change before a stable 1.0 release.
