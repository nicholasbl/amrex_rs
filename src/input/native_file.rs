use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};

use glam::prelude::*;

use super::{data_reader::DataReader, native_file_parse::*, text_parsing::*};

/// Physical-space bounding box.
#[derive(Debug)]
pub struct BoundingBox {
    /// Lower physical coordinate.
    pub min: DVec3,
    /// Upper physical coordinate.
    pub max: DVec3,
}

/// Integer AMReX box with inclusive upper bounds.
#[derive(Debug, Clone)]
pub struct IndexDomain {
    /// Inclusive lower index.
    pub min: IVec3,
    /// Inclusive upper index.
    pub max: IVec3, // inclusive
    /// AMReX index type: zero for cell-centered axes, one for nodal axes.
    pub index_type: IVec3,
}

/// Parsed top-level AMReX `Header` metadata.
#[derive(Debug)]
pub struct Header {
    /// Simulation time recorded in the plotfile.
    pub simulation_time: f64,
    /// Finest AMR level index present in the plotfile.
    pub finest_level: usize,
    /// Physical domain bounds.
    pub domain: BoundingBox,
    /// Variables stored in each FAB.
    pub variables: Vec<Variable>,
    /// Refinement ratios between adjacent levels.
    pub refinement_ratios: Vec<usize>,
    /// Per-level integer domains.
    pub index_domains: Vec<IndexDomain>,
    /// Per-level time step counters.
    pub level_steps: Vec<usize>,
    /// Per-level cell sizes.
    pub cell_sizes: Vec<DVec3>,
    /// Coordinate system declared by the plotfile.
    pub coordinate_system: CoordinateSystem,
    /// Plotfile boundary width.
    pub boundary_width: usize,
    /// Per-level records from the header.
    pub level_records: Vec<PerLevelRecord>,
}

/// Named component in the plotfile.
#[derive(Debug)]
pub struct Variable {
    /// Component name.
    pub name: String,
    /// Zero-based component index.
    pub index: usize,
}

/// AMReX coordinate-system identifier.
#[derive(Debug)]
pub enum CoordinateSystem {
    /// Cartesian coordinates.
    Cartesian,
    /// Cylindrical coordinates.
    Cylindrical,
    /// Spherical coordinates.
    Spherical,
}

/// Top-level metadata for one AMR level.
#[derive(Debug)]
pub struct PerLevelRecord {
    /// AMR level index.
    pub level_index: usize,
    /// Number of grid patches on the level.
    pub number_of_grid_patches: usize,
    /// Simulation time for this level.
    pub simulation_time: f64,
    /// Time step counter for this level.
    pub level_step: usize,
    /// Physical bounds of each grid patch.
    pub grids: Vec<BoundingBox>,
    /// Relative path prefix for the level's cell data.
    pub path: String,
}

/// A read-only view of an AMReX plotfile directory.
pub struct PlotFile {
    pub(crate) root: PathBuf,
    header: Header,
    cell_headers: Vec<OnceLock<CellHeader>>,
}

impl PlotFile {
    /// Open a plotfile and eagerly read its top-level `Header`.
    /// Per-level `Cell_H` metadata is loaded on first access.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let header = read_header(root.join("Header"))?;
        let cell_headers = (0..header.level_records.len())
            .map(|_| OnceLock::new())
            .collect();

        Ok(Self {
            root,
            header,
            cell_headers,
        })
    }

    /// Return the parsed top-level header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Return all variables in component order.
    pub fn variables(&self) -> &[Variable] {
        &self.header.variables
    }

    /// Find a variable by exact name.
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.header
            .variables
            .iter()
            .find(|variable| variable.name == name)
    }

    /// Return a level view by level index.
    pub fn level(&self, index: usize) -> Option<Level<'_>> {
        (index < self.header.level_records.len()).then_some(Level {
            plot_file: self,
            index,
        })
    }

    /// Iterate over all AMR levels.
    pub fn levels(&self) -> Levels<'_> {
        Levels {
            plot_file: self,
            next: 0,
        }
    }

    /// Create a data reader for FAB component data.
    pub fn data_reader(&self) -> DataReader<'_> {
        DataReader::new(self)
    }

    /// Load and cross-check every per-level `Cell_H` file.
    pub fn validate_all(&self) -> Result<()> {
        for level in self.levels() {
            let _ = level.patches()?;
        }
        Ok(())
    }

    fn cell_header(&self, level: usize) -> Result<&CellHeader> {
        let cache = self
            .cell_headers
            .get(level)
            .context("level index is out of range")?;

        if let Some(header) = cache.get() {
            return Ok(header);
        }

        let record = &self.header.level_records[level];
        let path = self.root.join(format!("{}_H", record.path));
        let parsed = read_hyper_cell_header(path)
            .with_context(|| format!("reading Cell_H metadata for level {level}"))?;
        validate_cell_header(&self.header, level, &parsed)?;

        // A concurrent reader may have populated the cache while this file was parsed.
        let _ = cache.set(parsed);
        Ok(cache.get().expect("Cell_H cache was just initialized"))
    }
}

/// Iterator over AMR levels in ascending level-index order.
pub struct Levels<'a> {
    plot_file: &'a PlotFile,
    next: usize,
}

impl<'a> Iterator for Levels<'a> {
    type Item = Level<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next;
        if index >= self.plot_file.header.level_records.len() {
            return None;
        }
        self.next += 1;
        Some(Level {
            plot_file: self.plot_file,
            index,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.plot_file.header.level_records.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Levels<'_> {}

/// Read-only view of one AMR level.
#[derive(Clone, Copy)]
pub struct Level<'a> {
    plot_file: &'a PlotFile,
    index: usize,
}

impl<'a> Level<'a> {
    /// Level index.
    pub fn index(self) -> usize {
        self.index
    }

    /// Simulation time for this level.
    pub fn time(self) -> f64 {
        self.record().simulation_time
    }

    /// Time step counter for this level.
    pub fn step(self) -> usize {
        self.record().level_step
    }

    /// Cell size for this level.
    pub fn cell_size(self) -> DVec3 {
        self.plot_file.header.cell_sizes[self.index]
    }

    /// Integer index domain for this level.
    pub fn index_domain(self) -> &'a IndexDomain {
        &self.plot_file.header.index_domains[self.index]
    }

    /// Number of patches on this level.
    pub fn patch_count(self) -> usize {
        self.record().number_of_grid_patches
    }

    /// Iterate over patches, loading and validating `Cell_H` metadata if needed.
    pub fn patches(self) -> Result<Patches<'a>> {
        let cell_header = self.plot_file.cell_header(self.index)?;
        Ok(Patches {
            plot_file: self.plot_file,
            level_index: self.index,
            physical_bounds: &self.record().grids,
            cell_header,
            next: 0,
        })
    }

    fn record(self) -> &'a PerLevelRecord {
        &self.plot_file.header.level_records[self.index]
    }
}

/// Iterator over patches on one AMR level.
pub struct Patches<'a> {
    plot_file: &'a PlotFile,
    level_index: usize,
    physical_bounds: &'a [BoundingBox],
    cell_header: &'a CellHeader,
    next: usize,
}

impl<'a> Iterator for Patches<'a> {
    type Item = Patch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let physical_bounds = self.physical_bounds.get(self.next)?;
        let fab = &self.cell_header.fabs[self.next];
        let index = self.next;
        self.next += 1;
        Some(Patch {
            plot_file: self.plot_file,
            level_index: self.level_index,
            index,
            physical_bounds,
            fab,
            bounds: &self.cell_header.bounds,
            ghost_cell_width: self.cell_header.ghost_cell_width,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.physical_bounds.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Patches<'_> {}

/// Read-only view of one AMReX FAB patch.
#[derive(Clone, Copy)]
pub struct Patch<'a> {
    pub(crate) plot_file: &'a PlotFile,
    pub(crate) level_index: usize,
    index: usize,
    physical_bounds: &'a BoundingBox,
    fab: &'a BoxInfo,
    bounds: &'a ComponentBounds,
    ghost_cell_width: usize,
}

impl<'a> Patch<'a> {
    /// Level index containing this patch.
    pub fn level_index(self) -> usize {
        self.level_index
    }
    /// Patch index within its level.
    pub fn index(self) -> usize {
        self.index
    }

    /// Physical-space patch bounds from the top-level header.
    pub fn physical_bounds(self) -> &'a BoundingBox {
        self.physical_bounds
    }

    /// Integer box covered by this patch, excluding ghost cells.
    pub fn index_box(self) -> &'a IndexDomain {
        &self.fab.box_info
    }

    /// Minimum value for a component on this patch, if present in `Cell_H`.
    pub fn component_min(self, component: usize) -> Option<f64> {
        self.bounds.get_minima(self.index, component)
    }

    /// Maximum value for a component on this patch, if present in `Cell_H`.
    pub fn component_max(self, component: usize) -> Option<f64> {
        self.bounds.get_maxima(self.index, component)
    }

    /// FAB data filename relative to the level directory.
    pub fn data_file(self) -> &'a str {
        &self.fab.file_name
    }

    /// Byte offset of this patch's FAB record in [`Patch::data_file`].
    pub fn data_offset(self) -> u64 {
        self.fab.file_offset
    }

    /// Number of ghost cells stored around this patch.
    pub fn ghost_cell_width(self) -> usize {
        self.ghost_cell_width
    }
}

// #[derive(Debug, Clone, Copy)]
// enum VisMfVersion {
//     VersionV1, // encoded as 1
// }

/// AMReX VisMF v1 storage mode.
#[derive(Debug, Clone, Copy)]
pub enum V1StorageMode {
    /// One data file per CPU/rank.
    OneFilePerCpu,
    /// A configured number of data files.
    NFiles,
}

/// Parsed `Cell_H` metadata for a level.
#[derive(Debug)]
pub struct CellHeader {
    /// Storage mode used by the level.
    pub mode: V1StorageMode,
    /// Ghost cell width present in FAB data.
    pub ghost_cell_width: usize,
    /// Per-patch FAB metadata.
    pub fabs: Vec<BoxInfo>,
    /// Per-component extrema.
    pub bounds: ComponentBounds,
}

/// File and box metadata for one FAB patch.
#[derive(Debug)]
pub struct BoxInfo {
    /// Integer box covered by the patch.
    pub box_info: IndexDomain,
    /// FAB data file name.
    pub file_name: Arc<str>,
    /// Byte offset of the FAB record.
    pub file_offset: u64,
}

/// Per-patch, per-component extrema from `Cell_H`.
#[derive(Debug)]
pub struct ComponentBounds {
    /// Number of patches represented.
    pub patch_count: usize,
    /// Number of components represented.
    pub component_count: usize,
    /// Row-major minima by patch then component.
    pub minima: Vec<f64>,
    /// Row-major maxima by patch then component.
    pub maxima: Vec<f64>,
}

impl ComponentBounds {
    pub(crate) fn read(
        lines: &mut Reader,
        expected_box_count: usize,
        expected_quant_count: usize,
    ) -> Result<ComponentBounds> {
        Ok(Self {
            patch_count: expected_box_count,
            component_count: expected_quant_count,
            minima: read_matrix(expected_box_count, expected_quant_count, lines)?,
            maxima: read_matrix(expected_box_count, expected_quant_count, lines)?,
        })
    }

    /// Return the stored minimum for a patch/component pair.
    pub fn get_minima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.minima
            .get(patch * self.component_count + component_id)
            .copied()
    }

    /// Return the stored maximum for a patch/component pair.
    pub fn get_maxima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.maxima
            .get(patch * self.component_count + component_id)
            .copied()
    }
}
