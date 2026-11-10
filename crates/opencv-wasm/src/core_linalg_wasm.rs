use wasm_bindgen::prelude::*;

use crate::{core_linalg, mat::Mat};

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = matGemmInto)]
pub fn mat_gemm_into(
    src1: &Mat,
    src2: &Mat,
    alpha: f64,
    src3: &Mat,
    beta: f64,
    destination: &Mat,
    flags: i32,
) -> Result<(), JsError> {
    core_linalg::gemm_into(src1, src2, alpha, src3, beta, destination, flags).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matEigenInto)]
pub fn mat_eigen_into(
    source: &Mat,
    eigenvalues: &Mat,
    eigenvectors: &Mat,
) -> Result<bool, JsError> {
    core_linalg::eigen_into(source, eigenvalues, eigenvectors).map_err(JsError::from)
}

#[wasm_bindgen(js_name = matSolvePolyInto)]
pub fn mat_solve_poly_into(
    coeffs: &Mat,
    roots: &Mat,
    #[wasm_bindgen(js_name = maxIters)] max_iters: i32,
) -> Result<i32, JsError> {
    core_linalg::solve_poly_into(coeffs, roots, max_iters).map_err(JsError::from)
}
