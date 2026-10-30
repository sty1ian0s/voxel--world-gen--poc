//! Logging setup shared by every binary in the workspace.
//!
//! Dev gets `debug` by default with the offending module inline; prod gets
//! `warn` unless `RUST_LOG` says otherwise. `log` + `env_logger` is used rather
//! than `tracing` because nothing here is async yet — one dependency fewer
//! until the streaming stage actually needs spans.

pub fn init(env: voxel_core::Env) {
    let default = if env.is_dev() { "debug" } else { "warn" };

    env_logger::Builder::from_env(
        // `RUST_LOG` from .env/.env.local wins; `default` is the fallback.
        env_logger::Env::default().default_filter_or(default),
    )
    .format_timestamp_millis()
    // Keep debug lines terse: `crate::module: message`.
    .format_target(env.is_dev())
    .init();
}
