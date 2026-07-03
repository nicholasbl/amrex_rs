use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};

use glam::prelude::*;

use crate::{native_file_parse::*, text_parsing::*};

#[derive(Debug)]
pub struct BoundingBox {
    pub min: DVec3,
    pub max: DVec3,
}

#[derive(Debug, Clone)]
pub struct IndexDomain {
    pub min: IVec3,
    pub max: IVec3, // inclusive
    pub index_type: IVec3,
}

#[derive(Debug)]
pub struct Header {
    pub simulation_time: f64,
    pub finest_level: usize,
    pub domain: BoundingBox,
    pub variables: Vec<Variable>,
    pub refinement_ratios: Vec<usize>,
    pub index_domains: Vec<IndexDomain>,
    pub level_steps: Vec<usize>,
    pub cell_sizes: Vec<DVec3>,
    pub coordinate_system: CoordinateSystem,
    pub boundary_width: usize,
    pub level_records: Vec<PerLevelRecord>,
}

#[derive(Debug)]
pub struct Variable {
    pub name: String,
    pub index: usize,
}

#[derive(Debug)]
pub enum CoordinateSystem {
    Cartesian,
    Cylindrical,
    Spherical,
}

#[derive(Debug)]
pub struct PerLevelRecord {
    pub level_index: usize,
    pub number_of_grid_patches: usize,
    pub simulation_time: f64,
    pub level_step: usize,
    pub grids: Vec<BoundingBox>,
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

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn variables(&self) -> &[Variable] {
        &self.header.variables
    }

    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.header
            .variables
            .iter()
            .find(|variable| variable.name == name)
    }

    pub fn level(&self, index: usize) -> Option<Level<'_>> {
        (index < self.header.level_records.len()).then_some(Level {
            plot_file: self,
            index,
        })
    }

    pub fn levels(&self) -> Levels<'_> {
        Levels {
            plot_file: self,
            next: 0,
        }
    }

    pub fn data_reader(&self) -> crate::DataReader<'_> {
        crate::DataReader::new(self)
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

#[derive(Clone, Copy)]
pub struct Level<'a> {
    plot_file: &'a PlotFile,
    index: usize,
}

impl<'a> Level<'a> {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn time(self) -> f64 {
        self.record().simulation_time
    }

    pub fn step(self) -> usize {
        self.record().level_step
    }

    pub fn cell_size(self) -> DVec3 {
        self.plot_file.header.cell_sizes[self.index]
    }

    pub fn index_domain(self) -> &'a IndexDomain {
        &self.plot_file.header.index_domains[self.index]
    }

    pub fn patch_count(self) -> usize {
        self.record().number_of_grid_patches
    }

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
    pub fn level_index(self) -> usize {
        self.level_index
    }
    pub fn index(self) -> usize {
        self.index
    }

    pub fn physical_bounds(self) -> &'a BoundingBox {
        self.physical_bounds
    }

    pub fn index_box(self) -> &'a IndexDomain {
        &self.fab.box_info
    }

    pub fn component_min(self, component: usize) -> Option<f64> {
        self.bounds.get_minima(self.index, component)
    }

    pub fn component_max(self, component: usize) -> Option<f64> {
        self.bounds.get_maxima(self.index, component)
    }

    pub fn data_file(self) -> &'a str {
        &self.fab.file_name
    }

    pub fn data_offset(self) -> u64 {
        self.fab.file_offset
    }

    pub fn ghost_cell_width(self) -> usize {
        self.ghost_cell_width
    }
}

// #[derive(Debug, Clone, Copy)]
// enum VisMfVersion {
//     VersionV1, // encoded as 1
// }

#[derive(Debug, Clone, Copy)]
pub enum V1StorageMode {
    OneFilePerCpu,
    NFiles,
}

#[derive(Debug)]
pub struct CellHeader {
    pub mode: V1StorageMode,
    pub ghost_cell_width: usize,
    pub fabs: Vec<BoxInfo>,
    pub bounds: ComponentBounds,
}

#[derive(Debug)]
pub struct BoxInfo {
    pub box_info: IndexDomain,
    pub file_name: Arc<str>,
    pub file_offset: u64,
}

#[derive(Debug)]
pub struct ComponentBounds {
    pub patch_count: usize,
    pub component_count: usize,
    pub minima: Vec<f64>,
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

    pub fn get_minima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.minima
            .get(patch * self.component_count + component_id)
            .copied()
    }

    pub fn get_maxima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.maxima
            .get(patch * self.component_count + component_id)
            .copied()
    }
}
