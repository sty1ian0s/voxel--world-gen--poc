mod logging;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    // Config first: it decides the log filter, and the filter must be set
    // before anything else can log.
    let cfg = voxel_core::Config::load().context("loading configuration")?;
    logging::init(cfg.env);

    // `warn` on purpose: prod's default filter is `warn`, and the resolved
    // config is the one thing an operator must always be able to see, even
    // when they cannot turn on debug logging.
    log::warn!("voxel-app starting: {}", cfg.summary());

    Ok(())
}
