from standard_imports import *
print('\033[93m\nImporting Standard Imports\033[0m')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler

import loess_rs
print('\033[93m\nPulled Rust connections for Loess\033[0m')

import warnings
warnings.filterwarnings('ignore')

print('\033[93mImporting LoessScaler\033[0m')
print('\033[93Importing fred()\033[0m')

def fred(series_id, start=None, end=None):
    """Pull a FRED series by ID and return as a Series indexed by date."""
    import pandas as pd
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.set_index("observation_date").iloc[:, 0]
    df.name = series_id
    if start: df = df.loc[start:]
    if end:   df = df.loc[:end]
    return df

class LoessScaler(BaseEstimator, TransformerMixin):
    """Supervised scaler: smooth nonlinear transform via local linear regression.

    Parameters
    ----------
    frac : float, default=0.15
        Fraction of training data used in each local window.
    bw_floor_frac : float or None, default=0.02
        Minimum bandwidth as a fraction of the clipped range (hi - lo).
        0.02 means the window is at least 2% of the variable's range.
    n_grid : int, default=500
        Number of grid points for LOESS evaluation.
    clip_quantiles : tuple, default=(0.01, 0.99)
        Percentile range for grid and clipping.
    """
    def __init__(self, frac=0.15, bw_floor_frac=0.02, n_grid=500, clip_quantiles=(0.01, 0.99)):
        self.frac = frac
        self.bw_floor_frac = bw_floor_frac
        self.n_grid = n_grid
        self.clip_quantiles = clip_quantiles

    def fit(self, X, y):
        from scipy.interpolate import CubicSpline
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.splines_ = []; self.ranges_ = []; self.bw_floors_ = []
        for j in range(X.shape[1]):
            mask = ~np.isnan(X[:, j]); xj, yj = X[mask, j], y[mask]
            lo = np.percentile(xj, self.clip_quantiles[0] * 100)
            hi = np.percentile(xj, self.clip_quantiles[1] * 100)
            self.ranges_.append((lo, hi))
            bw_floor = self.bw_floor_frac * (hi - lo) if self.bw_floor_frac else None
            self.bw_floors_.append(bw_floor)
            keep = (xj >= lo) & (xj <= hi)
            x_grid = np.linspace(lo, hi, self.n_grid)
            p_grid = loess_rs.loess_on_grid(xj[keep], yj[keep], x_grid, self.frac, bw_floor)
            self.splines_.append(CubicSpline(x_grid, p_grid))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        Xt = np.zeros_like(X)
        for j, (spl, (lo, hi)) in enumerate(zip(self.splines_, self.ranges_)):
            mask = ~np.isnan(X[:, j]); xj = np.clip(X[mask, j], lo, hi)
            Xt[mask, j] = np.clip(spl(xj), 0, 1); Xt[~mask, j] = np.nan
        return Xt

    def plot(self, X, y, col_names=None, n_bins=30, lower_pct=1, upper_pct=99):
        """Plot LOESS fit vs empirical binned default rates for each column.

        Parameters
        ----------
        X, y : array-like
        col_names : list of str or None
        n_bins : int, default=30
            Number of equal-count bins for the empirical benchmark.
        lower_pct, upper_pct : float, default 1 and 99
            Percentile cutoffs. Observations outside [lower_pct, upper_pct]
            are dropped before binning, and the LOESS curve is plotted only
            over the kept range. Pass (0, 100) to disable.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n_cols = X.shape[1]
        if col_names is None:
            col_names = [f'Feature {j}' for j in range(n_cols)]

        for j in range(n_cols):
            lo, hi = self.ranges_[j]
            spl = self.splines_[j]
            col = col_names[j]

            # Trim percentile tails; clip survivors to the spline's domain
            x_orig = X[:, j]
            p_low, p_high = np.percentile(x_orig, [lower_pct, upper_pct])
            keep = (x_orig >= p_low) & (x_orig <= p_high)
            x_raw  = np.clip(x_orig[keep], lo, hi)
            y_use  = y[keep]
            n_drop = len(x_orig) - len(x_raw)

            # LOESS curve over the kept range
            x_plot = np.linspace(p_low, p_high, 1000)
            p_plot = np.clip(spl(x_plot), 0, 1)

            # Equal-count bins on kept data
            bin_edges = np.unique(np.percentile(x_raw, np.linspace(0, 100, n_bins + 1)))
            nb = len(bin_edges) - 1
            bin_idx = np.clip(np.digitize(x_raw, bin_edges) - 1, 0, nb - 1)

            mids, rates, counts, ci_lo, ci_hi = [], [], [], [], []
            for b in range(nb):
                m = bin_idx == b; n_b = m.sum()
                if n_b >= 10:
                    r = y_use[m].mean(); se = np.sqrt(r * (1 - r) / n_b)
                    mids.append(np.median(x_raw[m]))   # data center, not bin-edge midpoint
                    rates.append(r); counts.append(n_b)
                    ci_lo.append(max(r - 1.96 * se, 0))
                    ci_hi.append(r + 1.96 * se)
            mids, rates, counts = np.array(mids), np.array(rates), np.array(counts)
            ci_lo, ci_hi = np.array(ci_lo), np.array(ci_hi)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.concatenate([mids, mids[::-1]]),
                y=np.concatenate([ci_hi, ci_lo[::-1]]),
                fill='toself', fillcolor='rgba(31,119,180,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI (binomial)', hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=mids, y=rates, mode='markers',
                marker=dict(size=5, color='black'),
                name=f'Empirical ({nb} equal-count bins)',
                hovertemplate='x=%{x:.3f}<br>PD=%{y:.4f}<extra></extra>'))
            fig.add_trace(go.Scatter(
                x=x_plot, y=p_plot, mode='lines',
                line=dict(color='firebrick', width=1.5), name='LOESS',
                hovertemplate='x=%{x:.3f}<br>PD=%{y:.4f}<extra></extra>'))

            fig.update_layout(
                title='LOESS vs Empirical Default Rates',
                yaxis_title='Probability of Default (PD)', xaxis_title=col,
                width=600, height=400, template='plotly_white',
                legend=dict(font=dict(size=10), orientation='h',
                            yanchor='bottom', y=1.02, xanchor='left', x=0))
            fig.show()

            # Diagnostics
            p_at_mids = np.clip(spl(mids), 0, 1)
            outside = (p_at_mids < ci_lo) | (p_at_mids > ci_hi)
            print(f"{col}: trim=[{p_low:.3g}, {p_high:.3g}], "
                  f"dropped {n_drop}/{len(x_orig)} ({n_drop/len(x_orig):.1%})")
            print(f"  Equal-count bins: {len(mids)} (≈{counts.mean():.0f} obs each)")
            print(f"  LOESS outside 95% CI: {outside.sum()} of {len(mids)} bins\n")

print('\033[93mImporting Loess2DScaler\n\033[0m')

class Loess2DScaler(BaseEstimator, TransformerMixin):
    """Supervised scaler: 2D LOESS surface mapping (x1, x2) → P(default).

    Takes exactly 2 input columns. Returns 1 column: the smoothed
    joint P(default) from the 2D surface.

    Parameters
    ----------
    frac : float, default=0.10
        Fraction of training data used in each local window.
    bw_floor_frac : float or None, default=0.05
        Minimum bandwidth as fraction of the diagonal of the
        standardized clipped range.
    n_grid : int, default=60
        Grid resolution per axis (n_grid × n_grid surface).
    clip_quantiles : tuple, default=(0.01, 0.99)
        Percentile range for clipping both variables.
    """
    def __init__(self, frac=0.10, bw_floor_frac=0.05, n_grid=60, clip_quantiles=(0.01, 0.99)):
        self.frac = frac
        self.bw_floor_frac = bw_floor_frac
        self.n_grid = n_grid
        self.clip_quantiles = clip_quantiles

    def fit(self, X, y):
        from scipy.interpolate import RegularGridInterpolator
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[1] != 2:
            raise ValueError(f"Loess2DScaler expects 2 columns, got {X.shape[1]}")

        lo1 = np.percentile(X[:, 0], self.clip_quantiles[0] * 100)
        hi1 = np.percentile(X[:, 0], self.clip_quantiles[1] * 100)
        lo2 = np.percentile(X[:, 1], self.clip_quantiles[0] * 100)
        hi2 = np.percentile(X[:, 1], self.clip_quantiles[1] * 100)
        self.ranges_ = [(lo1, hi1), (lo2, hi2)]

        keep = (X[:, 0] >= lo1) & (X[:, 0] <= hi1) & (X[:, 1] >= lo2) & (X[:, 1] <= hi2)
        x1, x2, yc = X[keep, 0], X[keep, 1], y[keep]

        self.std_ = np.array([x1.std(), x2.std()])
        self.std_[self.std_ == 0] = 1.0

        g1 = np.linspace(lo1, hi1, self.n_grid)
        g2 = np.linspace(lo2, hi2, self.n_grid)
        self.grid_axes_ = (g2, g1)

        range_s = np.sqrt((hi1-lo1)**2/self.std_[0]**2 + (hi2-lo2)**2/self.std_[1]**2)
        bw_floor = self.bw_floor_frac * range_s if self.bw_floor_frac else None

        p_surface = loess_rs.loess_2d_on_grid(
            x1, x2, yc, g1, g2,
            self.std_[0], self.std_[1],
            self.frac, bw_floor
        )
        P = p_surface.reshape(self.n_grid, self.n_grid)
        self.interp_ = RegularGridInterpolator(
            self.grid_axes_, P, method='linear', bounds_error=False, fill_value=None)
        self.n_features_in_ = 2
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        x1 = np.clip(X[:, 0], self.ranges_[0][0], self.ranges_[0][1])
        x2 = np.clip(X[:, 1], self.ranges_[1][0], self.ranges_[1][1])
        pts = np.column_stack([x2, x1])
        return np.clip(self.interp_(pts), 0, 1).reshape(-1, 1)
