//! Utilities for reading AMReX plotfiles and extracting meshes from sparse AMR data.
//!
//! The crate exposes three common workflows:
//! opening plotfile metadata with [`PlotFile`], reading component data with
//! [`DataReader`], and extracting MC33 isosurfaces with [`isosurface()`].

pub mod compact;
mod input;
pub mod isosurface;
pub(crate) mod sparse_amr;
pub mod utility;

pub use compact::{CompactOptions, CompactPlot, read_compact, write_compact};
pub use isosurface::{IsosurfaceOptions, Mesh3D, Sample, Surface, isosurface, isosurface_compact};

pub use input::{
    BoundingBox, BoxInfo, ByteOrder, CellHeader, ComponentBounds, ComponentView, CoordinateSystem,
    DataReader, Header, IndexDomain, Level, LevelComponents, Levels, Patch, Patches,
    PerLevelRecord, PlotFile, ScalarType, V1StorageMode, Values, Variable, read_header,
    read_hyper_cell_header,
};
