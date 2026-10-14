//! Utilities for reading AMReX plotfiles and extracting meshes from sparse AMR data.
//!
//! The crate exposes three common workflows:
//! opening plotfile metadata with [`PlotFile`], reading component data with
//! [`DataReader`], and extracting MC33 isosurfaces with [`isosurface()`].

pub mod compact;
pub mod decimate;
mod input;
pub mod isosurface;
pub mod relax;
pub mod slice;
pub(crate) mod sparse_amr;
pub mod utility;

#[allow(deprecated)]
pub use compact::read_compact_selected_unchecked;
pub use compact::{
    CompactArchiveView, CompactChunkCodec, CompactChunkCodecSelection, CompactChunkCodecStats,
    CompactComponentCodec, CompactComponentErrorTolerance, CompactComponentScalarType,
    CompactComponentStats, CompactErrorTolerance, CompactOptions, CompactPlot, CompactScalarType,
    CompactStageTimings, CompactWorkCounts, NonFinitePolicy, NonFiniteReplacement,
    TranscodeCompactOptions, TranscodeCompactResult, WriteCompactResult, read_compact,
    read_compact_selected, transcode_compact, view_compact, view_compact_unchecked, write_compact,
};
pub use decimate::{
    DecimateErrorMode, DecimateOptions, DecimateParallelOptions, DecimatePipelineOptions,
    DecimatePipelineResult, DecimatePipelineTimings, DecimateResult, DecimateTarget,
    DecimateTimings, decimate_mesh, decimate_mesh_pipeline,
};
pub use isosurface::{
    IsosurfaceOptions, Mesh3D, RemoveDegenerateTrianglesOptions, Sample, Surface, isosurface,
    isosurface_compact, remove_degenerate_triangles,
};
pub use relax::{VertexRelaxOptions, VertexRelaxResult, VertexRelaxTimings, relax_mesh_vertices};
pub use slice::{SliceAxis, SliceOptions, SlicePlane, slice, slice_compact};

pub use input::{
    BoundingBox, BoxInfo, ByteOrder, CellHeader, ComponentBounds, ComponentView, CoordinateSystem,
    DataReader, Header, IndexDomain, Level, LevelComponents, Levels, Patch, Patches,
    PerLevelRecord, PlotFile, ScalarType, V1StorageMode, Values, Variable, read_header,
    read_hyper_cell_header,
};
