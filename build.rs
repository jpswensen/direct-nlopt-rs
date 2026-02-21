// build.rs - Optionally compiles NLOPT DIRECT C source files for comparison testing.
// Feature-gated behind "nlopt-compare" to avoid requiring a C compiler for normal builds.

fn main() {
    #[cfg(feature = "nlopt-compare")]
    {
        build_nlopt_direct();
    }
}

#[cfg(feature = "nlopt-compare")]
fn build_nlopt_direct() {
    let nlopt_direct_dir = std::path::Path::new("../nlopt/src/algs/direct");
    let shim_dir = std::path::Path::new("nlopt-shim");

    // Verify source files exist
    assert!(
        nlopt_direct_dir.join("DIRect.c").exists(),
        "NLOPT DIRECT source not found at {:?}",
        nlopt_direct_dir
    );

    cc::Build::new()
        // NLOPT DIRECT algorithm source files
        .file(nlopt_direct_dir.join("DIRect.c"))
        .file(nlopt_direct_dir.join("DIRsubrout.c"))
        .file(nlopt_direct_dir.join("DIRserial.c"))
        .file(nlopt_direct_dir.join("direct_wrap.c"))
        // Minimal utility shim (provides nlopt_seconds, nlopt_stop_time_, etc.)
        .file(shim_dir.join("nlopt_util_shim.c"))
        // Include paths: shim headers first (provide nlopt_config.h, nlopt.h),
        // then NLOPT source directories for direct.h and direct-internal.h
        .include(shim_dir)
        .include(nlopt_direct_dir)
        // Compiler flags
        .opt_level(2)
        .warnings(false)
        .compile("nlopt_direct");
}
