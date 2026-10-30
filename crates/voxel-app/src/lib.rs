//! Shared front-end plumbing for the binaries in this crate.
//!
//! `voxel-app` builds a library as well as binaries so the desktop binary and
//! `bin/debug.rs` share camera, mesh conversion, GPU setup and logging instead
//! of each carrying a copy.

pub mod app;
pub mod camera;
pub mod font;
pub mod gpu;
pub mod hud;
pub mod logging;
pub mod render;
pub mod stats;
