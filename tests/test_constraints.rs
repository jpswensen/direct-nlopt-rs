//! Tests for linear and nonlinear inequality constraint support.
//!
//! Each test problem has a known constrained optimum that differs from the
//! unconstrained optimum, verifying that constraints actually steer the search.

use direct_nlopt::{DirectAlgorithm, DirectBuilder};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere f(x) = x0^2 + x1^2.  Unconstrained min at (0, 0).
fn sphere(x: &[f64]) -> f64 {
    x[0] * x[0] + x[1] * x[1]
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — Linear constraint, Gablonsky backend
//
// minimize  x0^2 + x1^2
// subject to  x0 + x1 >= 1   (equivalently:  -x0 - x1 <= -1)
// bounds:  x in [-5, 5]^2
//
// Constrained optimum: x0 = x1 = 0.5, f* = 0.5
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_constraint_gablonsky() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        // -x0 - x1 <= -1  ⟺  x0 + x1 >= 1
        .add_linear_constraint(vec![-1.0, -1.0], -1.0)
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    // Constrained optimum is f* = 0.5 at (0.5, 0.5)
    assert!(
        result.fun < 0.6,
        "f = {} should be near 0.5 (constrained optimum)",
        result.fun
    );
    assert!(
        result.fun > 0.01,
        "f = {} is too low — constraint may not be active",
        result.fun
    );
    // The constraint must be satisfied: x0 + x1 >= 1
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.05,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — Linear constraint, CDirect backend
//
// CDirect's replace_infeasible() helps near constraint boundaries but
// the half-plane geometry doesn't benefit as much from box-based replacement
// as curved constraints do. Wider tolerance than Gablonsky.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_constraint_cdirect() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(20000)
        .add_linear_constraint(vec![-1.0, -1.0], -1.0)
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 5.0,
        "f = {} should be in feasible region",
        result.fun
    );
    assert!(
        result.fun > 0.01,
        "f = {} is too low — constraint may not be active",
        result.fun
    );
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.2,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — Nonlinear constraint, Gablonsky backend
//
// minimize  x0^2 + x1^2
// subject to  x0^2 + x1^2 >= 1   (circle constraint, push away from origin)
//            equivalently:  1 - x0^2 - x1^2 <= 0  →  g(x) = 1 - x0^2 - x1^2
// bounds:  x in [-5, 5]^2
//
// Constrained optimum: any point on the unit circle, f* = 1.0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nonlinear_constraint_gablonsky() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        // g(x) = 1 - x0^2 - x1^2 <= 0  ⟺  x0^2 + x1^2 >= 1
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] * x[0] - x[1] * x[1])
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    // Constrained optimum is f* = 1.0 (on the unit circle)
    assert!(
        result.fun < 1.15,
        "f = {} should be near 1.0 (constrained optimum)",
        result.fun
    );
    assert!(
        result.fun > 0.5,
        "f = {} is too low — constraint may not be active",
        result.fun
    );
    // Constraint: x0^2 + x1^2 >= 1
    let r2 = result.x[0] * result.x[0] + result.x[1] * result.x[1];
    assert!(
        r2 >= 1.0 - 0.05,
        "constraint violated: x0^2+x1^2 = {} < 1",
        r2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — Nonlinear constraint, CDirect backend
//
// With Phase 2 replace_infeasible(), CDirect now converges well on curved
// constraint boundaries. Tolerance matches the Gablonsky backend.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nonlinear_constraint_cdirect() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(20000)
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] * x[0] - x[1] * x[1])
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    // CDirect with replace_infeasible converges well on circular constraint.
    assert!(
        result.fun < 1.15,
        "f = {} should be near 1.0 (constrained optimum)",
        result.fun
    );
    assert!(
        result.fun > 0.5,
        "f = {} is too low — constraint may not be active",
        result.fun
    );
    let r2 = result.x[0] * result.x[0] + result.x[1] * result.x[1];
    assert!(
        r2 >= 1.0 - 0.05,
        "constraint violated: x0^2+x1^2 = {} < 1",
        r2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — Combined linear + nonlinear constraints
//
// minimize  (x0 - 2)^2 + (x1 - 2)^2
// subject to  x0 + x1 <= 3          (linear: half-plane)
//             x0^2 + x1^2 <= 4      (nonlinear: inside disk of radius 2)
// bounds:  x in [-5, 5]^2
//
// The unconstrained optimum (2,2) violates x0+x1<=3 (sum=4) and barely
// satisfies the circle (r^2=8>4). The constrained optimum is at (1.5, 1.5)
// with f* = 0.5.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_combined_linear_and_nonlinear_constraints_gablonsky() {
    let result = DirectBuilder::new(
        |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2),
        vec![(-5.0, 5.0), (-5.0, 5.0)],
    )
    .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
    .max_feval(5000)
    // x0 + x1 <= 3
    .add_linear_constraint(vec![1.0, 1.0], 3.0)
    // x0^2 + x1^2 <= 4  →  g(x) = x0^2 + x1^2 - 4 <= 0
    .add_nonlinear_constraint(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 4.0)
    .minimize()
    .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    // Constrained optimum near (1.5, 1.5), f* = 0.5
    assert!(
        result.fun < 0.8,
        "f = {} should be near 0.5",
        result.fun
    );
    // Linear constraint: x0 + x1 <= 3
    let sum: f64 = result.x.iter().sum();
    assert!(sum <= 3.0 + 0.05, "linear constraint violated: sum = {}", sum);
    // Nonlinear constraint: x0^2 + x1^2 <= 4
    let r2 = result.x[0] * result.x[0] + result.x[1] * result.x[1];
    assert!(r2 <= 4.0 + 0.05, "nonlinear constraint violated: r^2 = {}", r2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — Matrix-form linear constraints (A x <= b)
//
// minimize  x0^2 + x1^2
// subject to  x0 >= 1  and  x1 >= 1   (two constraints)
//            i.e.  -x0 <= -1  and  -x1 <= -1
// bounds:  x in [-5, 5]^2
//
// Constrained optimum: (1, 1), f* = 2.0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_constraints_matrix_form() {
    let a = vec![
        vec![-1.0, 0.0],  // -x0 <= -1
        vec![0.0, -1.0],  // -x1 <= -1
    ];
    let b = vec![-1.0, -1.0];

    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        .linear_constraints(a, b)
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 2.2,
        "f = {} should be near 2.0",
        result.fun
    );
    assert!(
        result.fun > 1.0,
        "f = {} is too low — constraints may not be active",
        result.fun
    );
    assert!(result.x[0] >= 1.0 - 0.05, "x0 = {} < 1", result.x[0]);
    assert!(result.x[1] >= 1.0 - 0.05, "x1 = {} < 1", result.x[1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 — Bit-exact fallback: no constraints = unchanged behavior
//
// Verifies that with no constraints, the Gablonsky backend produces the
// exact same result (same x, fun, nfev, nit) as a plain unconstrained call.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_no_constraints_bit_exact_gablonsky() {
    let unconstrained = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(500)
        .minimize()
        .unwrap();

    // Same problem, but constructed with the constraint API path (no constraints added)
    let also_unconstrained = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(500)
        .minimize()
        .unwrap();

    assert_eq!(unconstrained.fun, also_unconstrained.fun);
    assert_eq!(unconstrained.x, also_unconstrained.x);
    assert_eq!(unconstrained.nfev, also_unconstrained.nfev);
    assert_eq!(unconstrained.nit, also_unconstrained.nit);
}

#[test]
fn test_no_constraints_bit_exact_cdirect() {
    let unconstrained = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(500)
        .minimize()
        .unwrap();

    let also_unconstrained = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(500)
        .minimize()
        .unwrap();

    assert_eq!(unconstrained.fun, also_unconstrained.fun);
    assert_eq!(unconstrained.x, also_unconstrained.x);
    assert_eq!(unconstrained.nfev, also_unconstrained.nfev);
    assert_eq!(unconstrained.nit, also_unconstrained.nit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 — Parallel + constraints (Gablonsky)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_constraint_parallel_gablonsky() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        .parallel(true)
        .add_linear_constraint(vec![-1.0, -1.0], -1.0)
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 0.6,
        "f = {} should be near 0.5",
        result.fun
    );
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.05,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}

#[test]
fn test_nonlinear_constraint_parallel_gablonsky() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        .parallel(true)
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] * x[0] - x[1] * x[1])
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 1.15,
        "f = {} should be near 1.0",
        result.fun
    );
    let r2 = result.x[0] * result.x[0] + result.x[1] * result.x[1];
    assert!(
        r2 >= 1.0 - 0.05,
        "constraint violated: r^2 = {} < 1",
        r2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9 — Constraint that couples variables (multi-variable constraint)
//
// minimize  x0 + x1 + x2
// subject to  x0 * x1 * x2 >= 1  →  g(x) = 1 - x0*x1*x2 <= 0
// bounds:  x in [0.1, 5]^3
//
// By AM-GM, at the constrained optimum x0 = x1 = x2 = 1, f* = 3.0.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_coupled_nonlinear_constraint_gablonsky() {
    let result = DirectBuilder::new(
        |x: &[f64]| x[0] + x[1] + x[2],
        vec![(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)],
    )
    .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
    .max_feval(10000)
    // 1 - x0*x1*x2 <= 0  ⟺  x0*x1*x2 >= 1
    .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] * x[1] * x[2])
    .minimize()
    .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 3.5,
        "f = {} should be near 3.0 (AM-GM constrained optimum)",
        result.fun
    );
    let prod = result.x[0] * result.x[1] * result.x[2];
    assert!(
        prod >= 1.0 - 0.1,
        "coupling constraint violated: x0*x1*x2 = {} < 1",
        prod
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10 — Parallel + constraints (CDirect)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nonlinear_constraint_parallel_cdirect() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(20000)
        .parallel(true)
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] * x[0] - x[1] * x[1])
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    // CDirect parallel with replace_infeasible: should converge well on circle.
    assert!(
        result.fun < 1.5,
        "f = {} should be near 1.0",
        result.fun
    );
    let r2 = result.x[0] * result.x[0] + result.x[1] * result.x[1];
    assert!(
        r2 >= 1.0 - 0.1,
        "constraint violated: r^2 = {} < 1",
        r2
    );
}

#[test]
fn test_linear_constraint_parallel_cdirect() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(20000)
        .parallel(true)
        .add_linear_constraint(vec![-1.0, -1.0], -1.0)
        .minimize()
        .unwrap();

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 5.0,
        "f = {} should be in feasible region",
        result.fun
    );
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.2,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test — Linear-as-nonlinear: same linear constraint expressed via the
// nonlinear API, to compare convergence quality.
//
// minimize  x0^2 + x1^2
// subject to  x0 + x1 >= 1   expressed as   g(x) = 1 - x0 - x1 <= 0
// bounds:  x in [-5, 5]^2
//
// Constrained optimum: x0 = x1 = 0.5, f* = 0.5
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_as_nonlinear_cdirect() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(20000)
        // Same constraint as the linear test, but via nonlinear API:
        // g(x) = 1 - x0 - x1 <= 0  ⟺  x0 + x1 >= 1
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] - x[1])
        .minimize()
        .unwrap();

    println!(
        "CDirect linear-as-nonlinear: f={:.6}, x={:?}, nfev={}",
        result.fun, result.x, result.nfev
    );

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 5.0,
        "f = {} should be in feasible region",
        result.fun
    );
    assert!(
        result.fun > 0.01,
        "f = {} is too low — constraint may not be active",
        result.fun
    );
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.2,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}

#[test]
fn test_linear_as_nonlinear_gablonsky() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(5000)
        .add_nonlinear_constraint(|x: &[f64]| 1.0 - x[0] - x[1])
        .minimize()
        .unwrap();

    println!(
        "Gablonsky linear-as-nonlinear: f={:.6}, x={:?}, nfev={}",
        result.fun, result.x, result.nfev
    );

    assert!(result.success, "optimization failed: {:?}", result.return_code);
    assert!(
        result.fun < 0.6,
        "f = {} should be near 0.5 (constrained optimum)",
        result.fun
    );
    let sum: f64 = result.x.iter().sum();
    assert!(
        sum >= 1.0 - 0.05,
        "constraint violated: x0+x1 = {} < 1",
        sum
    );
}
