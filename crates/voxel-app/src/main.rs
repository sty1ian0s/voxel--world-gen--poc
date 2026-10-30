//! Desktop entry point: parse config, set up logging, run the window.

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    // Config first: it selects the log filter, and the filter must be set
    // before anything else can log.
    let cfg = voxel_core::Config::load().context("loading configuration")?;
    voxel_app::logging::init(cfg.env);

    // `warn` on purpose: prod's default filter is `warn`, and the resolved
    // config is the one thing an operator must always be able to see.
    log::warn!("voxel-app starting: {}", cfg.summary());

    voxel_app::app::run(cfg)
}
