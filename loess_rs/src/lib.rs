use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rayon::prelude::*;

/// 1D local linear regression on a grid — Rust replacement for _loess_on_grid.
///
/// For each grid point, finds the k-th nearest training point to set bandwidth,
/// computes tricube weights, solves the 2x2 weighted least squares, and returns
/// the intercept (smoothed P(default)).
///
/// Returns a numpy array of length len(x_grid), clipped to [0, 1].
#[pyfunction]
#[pyo3(signature = (x_train, y_train, x_grid, frac, bw_floor=None))]
fn loess_on_grid<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray1<'py, f64>,
    y_train: PyReadonlyArray1<'py, f64>,
    x_grid:  PyReadonlyArray1<'py, f64>,
    frac: f64,
    bw_floor: Option<f64>,
) -> Bound<'py, PyArray1<f64>> {

    // Copy data out of numpy arrays so we can send to rayon threads
    let x: Vec<f64> = x_train.as_slice().unwrap().to_vec();
    let y: Vec<f64> = y_train.as_slice().unwrap().to_vec();
    let grid: Vec<f64> = x_grid.as_slice().unwrap().to_vec();

    let n = x.len();
    let k = ((frac * n as f64) as usize).max(50).min(n - 1);

    // Parallel map over grid points
    let result: Vec<f64> = grid.par_iter().map(|&xg| {

        // 1. Distances from this grid point to all training points
        let mut dists: Vec<f64> = x.iter().map(|&xi| (xg - xi).abs()).collect();

        // 2. Find k-th smallest distance (partial sort, like np.partition)
        dists.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        let mut h = dists[k] * 1.001;
        if let Some(bf) = bw_floor {
            h = h.max(bf);
        }

        // 3. Weighted local linear regression
        //    We accumulate the 5 sums needed for the 2x2 normal equations:
        //    [sw   swx ] [b0]   [swy ]
        //    [swx  swxx] [b1] = [swxy]
        //    and return b0 (the intercept = smoothed value at xg).
        let mut sw   = 0.0_f64;
        let mut swx  = 0.0_f64;
        let mut swxx = 0.0_f64;
        let mut swy  = 0.0_f64;
        let mut swxy = 0.0_f64;

        for i in 0..n {
            let d = (x[i] - xg).abs();
            let u = d / h;
            if u < 1.0 {
                let u3 = u * u * u;
                let w = (1.0 - u3) * (1.0 - u3) * (1.0 - u3);  // tricube
                let xi = x[i] - xg;
                sw   += w;
                swx  += w * xi;
                swxx += w * xi * xi;
                swy  += w * y[i];
                swxy += w * xi * y[i];
            }
        }

        // 4. Solve for intercept
        let det = sw * swxx - swx * swx;
        let p = if det.abs() > 1e-12 {
            (swxx * swy - swx * swxy) / det
        } else if sw > 0.0 {
            swy / sw
        } else {
            0.0
        };

        p.clamp(0.0, 1.0)
    }).collect();

    result.into_pyarray_bound(py)
}

/// 2D local linear regression on a grid — Rust replacement for Loess2DScaler's fit loop.
///
/// Takes two training variables (already clipped) and their standard deviations,
/// fits a local linear plane at each point on an (n_g1 × n_g2) grid, and returns
/// the smoothed surface as a flat array in row-major order (g2 varies slow, g1 fast),
/// matching the Python meshgrid(g1, g2) layout.
///
/// The 3×3 normal equations at each grid point are:
///   [sw    sw1   sw2 ] [b0]   [swy ]
///   [sw1   sw11  sw12] [b1] = [sw1y]
///   [sw2   sw12  sw22] [b2]   [sw2y]
/// We solve via Cramer's rule and return b0 (the intercept).
#[pyfunction]
#[pyo3(signature = (x1_train, x2_train, y_train, g1, g2, std1, std2, frac, bw_floor=None))]
fn loess_2d_on_grid<'py>(
    py: Python<'py>,
    x1_train: PyReadonlyArray1<'py, f64>,
    x2_train: PyReadonlyArray1<'py, f64>,
    y_train:  PyReadonlyArray1<'py, f64>,
    g1:       PyReadonlyArray1<'py, f64>,   // grid axis 1
    g2:       PyReadonlyArray1<'py, f64>,   // grid axis 2
    std1: f64,
    std2: f64,
    frac: f64,
    bw_floor: Option<f64>,
) -> Bound<'py, PyArray1<f64>> {

    let x1: Vec<f64> = x1_train.as_slice().unwrap().to_vec();
    let x2: Vec<f64> = x2_train.as_slice().unwrap().to_vec();
    let y:  Vec<f64> = y_train.as_slice().unwrap().to_vec();
    let grid1: Vec<f64> = g1.as_slice().unwrap().to_vec();
    let grid2: Vec<f64> = g2.as_slice().unwrap().to_vec();

    let n = x1.len();
    let k = ((frac * n as f64) as usize).max(100).min(n - 1);

    // Pre-standardize training points for distance calculation
    let x1s: Vec<f64> = x1.iter().map(|&v| v / std1).collect();
    let x2s: Vec<f64> = x2.iter().map(|&v| v / std2).collect();

    // Build flat grid: for each (g2_val, g1_val) — row-major, g2 slow, g1 fast
    let n_g1 = grid1.len();
    let n_g2 = grid2.len();
    let mut grid_pts: Vec<(f64, f64)> = Vec::with_capacity(n_g2 * n_g1);
    for &g2v in &grid2 {
        for &g1v in &grid1 {
            grid_pts.push((g1v, g2v));
        }
    }

    let result: Vec<f64> = grid_pts.par_iter().map(|&(gp1, gp2)| {
        let gp1s = gp1 / std1;
        let gp2s = gp2 / std2;

        // 1. Euclidean distances in standardized space
        let mut dists: Vec<f64> = (0..n).map(|i| {
            let d1 = gp1s - x1s[i];
            let d2 = gp2s - x2s[i];
            (d1 * d1 + d2 * d2).sqrt()
        }).collect();

        // 2. Adaptive bandwidth from k-th nearest neighbor
        dists.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        let mut h = dists[k] * 1.001;
        if let Some(bf) = bw_floor {
            h = h.max(bf);
        }

        // 3. Accumulate 3×3 normal equation sums
        let mut sw   = 0.0_f64;
        let mut sw1  = 0.0_f64;  let mut sw2  = 0.0_f64;
        let mut sw11 = 0.0_f64;  let mut sw22 = 0.0_f64;  let mut sw12 = 0.0_f64;
        let mut swy  = 0.0_f64;
        let mut sw1y = 0.0_f64;  let mut sw2y = 0.0_f64;

        for i in 0..n {
            let d1 = gp1s - x1s[i];
            let d2 = gp2s - x2s[i];
            let dist = (d1 * d1 + d2 * d2).sqrt();
            let u = dist / h;
            if u < 1.0 {
                let u3 = u * u * u;
                let w = (1.0 - u3) * (1.0 - u3) * (1.0 - u3);
                let xi1 = x1[i] - gp1;   // deviations in original scale
                let xi2 = x2[i] - gp2;
                sw   += w;
                sw1  += w * xi1;
                sw2  += w * xi2;
                sw11 += w * xi1 * xi1;
                sw22 += w * xi2 * xi2;
                sw12 += w * xi1 * xi2;
                swy  += w * y[i];
                sw1y += w * xi1 * y[i];
                sw2y += w * xi2 * y[i];
            }
        }

        // 4. Solve 3×3 via Cramer's rule for b0
        let det = sw  * (sw11 * sw22 - sw12 * sw12)
                - sw1 * (sw1  * sw22 - sw12 * sw2)
                + sw2 * (sw1  * sw12 - sw11 * sw2);

        let p = if det.abs() > 1e-12 {
            let num = swy  * (sw11 * sw22 - sw12 * sw12)
                    - sw1y * (sw1  * sw22 - sw12 * sw2)
                    + sw2y * (sw1  * sw12 - sw11 * sw2);
            num / det
        } else if sw > 0.0 {
            swy / sw
        } else {
            0.0
        };

        p.clamp(0.0, 1.0)
    }).collect();

    result.into_pyarray_bound(py)
}

/// Python module definition
#[pymodule]
fn loess_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loess_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(loess_2d_on_grid, m)?)?;
    Ok(())
}
