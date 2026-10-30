//! Shared front-end plumbing for the binaries in this crate.
//!
//! `voxel-app` builds a library as well as binaries so that `main.rs` and
//! `bin/debug.rs` share logging setup instead of each carrying a copy.

pub mod logging;
