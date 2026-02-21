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
    let nlopt_cdirect_dir = std::path::Path::new("../nlopt/src/algs/cdirect");
    let nlopt_util_dir = std::path::Path::new("../nlopt/src/util");
    let shim_dir = std::path::Path::new("nlopt-shim");

    // Verify source files exist
    assert!(
        nlopt_direct_dir.join("DIRect.c").exists(),
        "NLOPT DIRECT source not found at {:?}",
        nlopt_direct_dir
    );
    assert!(
        nlopt_cdirect_dir.join("cdirect.c").exists(),
        "NLOPT cdirect source not found at {:?}",
        nlopt_cdirect_dir
    );

    cc::Build::new()
        // NLOPT DIRECT algorithm source files (Gablonsky translation)
        .file(nlopt_direct_dir.join("DIRect.c"))
        .file(nlopt_direct_dir.join("DIRsubrout.c"))
        .file(nlopt_direct_dir.join("DIRserial.c"))
        .file(nlopt_direct_dir.join("direct_wrap.c"))
        // NLOPT cdirect algorithm source files (SGJ re-implementation)
        .file(nlopt_cdirect_dir.join("cdirect.c"))
        // NLOPT utility source files (stop, redblack tree, qsort_r)
        .file(nlopt_util_dir.join("stop.c"))
        .file(nlopt_util_dir.join("redblack.c"))
        .file(nlopt_util_dir.join("qsort_r.c"))
        // Shim providing additional test helpers (thirds, levels, hull, diameter)
        .file(shim_dir.join("nlopt_util_shim.c"))
        // Trace shim providing step-by-step tracing wrapper for DIRECT
        .file(shim_dir.join("nlopt_trace_shim.c"))
        // Include paths: shim headers first (provide nlopt_config.h, nlopt.h),
        // then NLOPT source directories
        .include(shim_dir)
        .include(nlopt_direct_dir)
        .include(nlopt_cdirect_dir)
        .include(nlopt_util_dir)
        // Compiler flags
        .opt_level(2)
        .warnings(false)
        .compile("nlopt_direct");

    // Ensure rebuild when shim sources change
    println!("cargo:rerun-if-changed=nlopt-shim/nlopt_trace_shim.c");
    println!("cargo:rerun-if-changed=nlopt-shim/nlopt_util_shim.c");
    println!("cargo:rerun-if-changed=nlopt-shim/direct-internal.h");
    println!("cargo:rerun-if-changed=build.rs");
}
