//! AMReX plotfile reading and conversion into multiresolution sparse grids.
//!
//! `amrex_rs` owns the AMReX input format only. The source-independent MSG
//! model, `.msg` archive I/O, isosurfacing, slicing, and mesh processing live in
//! `multires_rs`.

mod input;
mod msg;

pub use input::{
    BoundingBox, BoxInfo, ByteOrder, CellHeader, ComponentBounds, ComponentView, CoordinateSystem,
    DataReader, Header, IndexDomain, Level, LevelComponents, Levels, Patch, Patches,
    PerLevelRecord, PlotFile, ScalarType, V1StorageMode, Values, Variable, read_header,
    read_hyper_cell_header,
};
pub use msg::{
    AmrexInvalidValuePolicy, AmrexMsgField, AmrexToMsgOptions, AmrexToMsgReport, plotfile_to_msg,
};
pub use multires_rs::{MsgDataset, MsgScalarType};
