"""
pc_matching.visualizer
======================
Publication-quality diagnostic visualizations for Static-Dynamic Sw Matching
and Capillary Pressure Optimization.
Supports multi-stage comparison (Baseline vs. Multiplier-Only vs. Endpoint-Optimized).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from .scal_manager import SCALManager


def set_plotting_style():
    """Configure modern clean aesthetics."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 120
    })


def plot_scal_comparison(
    scal_manager: SCALManager,
    multipliers_dict: Optional[Dict[int, float]] = None,
    endpoint_params_dict: Optional[Dict[int, Any]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot base vs. multiplier-only vs. endpoint-optimized Capillary Pressure curves for all rock types.
    """
    set_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, rt_id in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        if rt_id not in scal_manager.tables:
            ax.set_title(f"Rock Type {rt_id} (Not Available)")
            continue

        table = scal_manager.tables[rt_id]
        base_sw = table.base_sw
        base_swirr = table.base_swirr
        base_pc = table.base_pcow
        s_wn = (base_sw - base_swirr) / (1.0 - base_swirr)

        # 1. Base curve
        ax.plot(base_sw, base_pc, 'k--', lw=2.0, label=f'Base SCAL (Swirr={base_swirr:.2f})')

        # 2. Multiplier-only curve
        if multipliers_dict and rt_id in multipliers_dict:
            mult = float(multipliers_dict[rt_id])
            ax.plot(base_sw, base_pc * mult, 'b:', lw=2.2, label=f'Multiplier-Only (alpha={mult:.3f})')

        # 3. Endpoint & Curvature optimized curve
        if endpoint_params_dict and rt_id in endpoint_params_dict:
            p = endpoint_params_dict[rt_id]
            a = float(p.get('alpha', 1.0)) if isinstance(p, dict) else float(p[0])
            sw_val = float(p.get('swirr', base_swirr)) if isinstance(p, dict) else float(p[1])
            pce = float(p.get('pce', 0.0)) if isinstance(p, dict) else (float(p[2]) if len(p) > 2 else 0.0)
            beta = float(p.get('beta', 1.0)) if isinstance(p, dict) else (float(p[3]) if len(p) > 3 else 1.0)

            s_wn_curv = s_wn ** beta
            opt_sw = sw_val + s_wn_curv * (1.0 - sw_val)
            opt_pc = base_pc * a + pce * (1.0 - s_wn_curv)
            opt_pc[-1] = 0.0
            opt_pc = np.maximum(0.0, opt_pc)
            for i in range(len(opt_pc) - 2, -1, -1):
                if opt_pc[i] < opt_pc[i + 1]:
                    opt_pc[i] = opt_pc[i + 1]

            lbl = f'Opt (alpha={a:.2f}, Swirr={sw_val:.3f}, Pce={pce:.2f}, beta={beta:.2f})'
            ax.plot(opt_sw, opt_pc, 'r-', lw=2.5, label=lbl)

        ax.set_title(f"Rock Type {rt_id}", fontweight='bold')
        ax.set_xlabel("Water Saturation Sw [frac]")
        ax.set_ylabel("Capillary Pressure Pcow [bar]")
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True, loc='upper right', fontsize=8)

    plt.suptitle("SCAL Capillary Pressure Curves: Baseline vs. Multiplier-Only vs. Endpoint & Curvature (Beta)", fontweight='bold')
    plt.tight_layout()
    return fig


def plot_saturation_height(
    depth_active: np.ndarray,
    rt_active: np.ndarray,
    sw_static: np.ndarray,
    sw_dynamic_init: np.ndarray,
    sw_dynamic_opt: np.ndarray,
    sw_dynamic_mult: Optional[np.ndarray] = None,
    fwl_depth: float = 925.0,
    sample_size: int = 5000,
    figsize: Tuple[int, int] = (16, 5)
) -> plt.Figure:
    """
    Plot Saturation vs. Depth profiles for Static, Baseline Dynamic, Multiplier-only, and Endpoint-Optimized.
    """
    set_plotting_style()
    unique_rts = [r for r in [1, 2, 3] if r in np.unique(rt_active)]
    n_plots = len(unique_rts)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]

    for idx, rt in enumerate(unique_rts):
        ax = axes[idx]
        mask = (rt_active == rt)
        d_rt = depth_active[mask]
        sw_stat_rt = sw_static[mask]
        sw_init_rt = sw_dynamic_init[mask]
        sw_opt_rt = sw_dynamic_opt[mask]
        sw_mult_rt = sw_dynamic_mult[mask] if sw_dynamic_mult is not None else None

        n_cells = len(d_rt)
        if n_cells > sample_size:
            sample_idx = np.random.choice(n_cells, sample_size, replace=False)
            d_rt = d_rt[sample_idx]
            sw_stat_rt = sw_stat_rt[sample_idx]
            sw_init_rt = sw_init_rt[sample_idx]
            sw_opt_rt = sw_opt_rt[sample_idx]
            if sw_mult_rt is not None:
                sw_mult_rt = sw_mult_rt[sample_idx]

        ax.scatter(sw_stat_rt, d_rt, s=8, alpha=0.3, c='blue', label='Static Sw (Petrel)', rasterized=True)
        ax.scatter(sw_init_rt, d_rt, s=8, alpha=0.3, c='black', label='Initial Dynamic (alpha=1.0)', rasterized=True)
        if sw_mult_rt is not None:
            ax.scatter(sw_mult_rt, d_rt, s=8, alpha=0.35, c='cyan', label='Stage 1: Multiplier-Only', rasterized=True)
        ax.scatter(sw_opt_rt, d_rt, s=8, alpha=0.45, c='red', label='Stage 2: Multiplier + Endpoint', rasterized=True)

        if fwl_depth is not None:
            ax.axhline(fwl_depth, color='darkblue', linestyle=':', lw=1.5, label=f'FWL ({fwl_depth}m)')
        ax.invert_yaxis()
        ax.set_title(f"Rock Type {rt} (n={np.sum(mask):,})", fontweight='bold')
        ax.set_xlabel("Water Saturation Sw [frac]")
        if idx == 0:
            ax.set_ylabel("True Vertical Depth [m TVD]")
        ax.set_xlim([0.0, 1.05])
        ax.legend(frameon=True, loc='upper left', markerscale=2.0, fontsize=8)

    plt.suptitle("Saturation vs. Depth Distribution: Static vs. Multiplier vs. Endpoint Optimized", fontweight='bold')
    plt.tight_layout()
    return fig


def plot_crossplots(
    sw_static: np.ndarray,
    sw_dynamic_init: np.ndarray,
    sw_dynamic_opt: np.ndarray,
    sw_dynamic_mult: Optional[np.ndarray] = None,
    rt_active: Optional[np.ndarray] = None,
    sample_size: int = 10000,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Cross-plot Sw_static vs. Sw_dynamic showing step-by-step progress across stages.
    """
    set_plotting_style()

    if sw_dynamic_mult is not None:
        if figsize is None:
            figsize = (18, 5)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        cases = [
            (axes[0], sw_dynamic_init, "Baseline Dynamic (alpha=1.0)", "Blues"),
            (axes[1], sw_dynamic_mult, "Stage 1: Multiplier-Only", "Purples"),
            (axes[2], sw_dynamic_opt, "Stage 2: Multiplier + Endpoint", "Reds")
        ]
    else:
        if figsize is None:
            figsize = (13, 6)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        cases = [
            (axes[0], sw_dynamic_init, "Baseline Dynamic (alpha=1.0)", "Blues"),
            (axes[1], sw_dynamic_opt, "Optimized Dynamic Sw", "Reds")
        ]

    n_pts = len(sw_static)
    idx = np.arange(n_pts)
    if n_pts > sample_size:
        idx = np.random.choice(n_pts, sample_size, replace=False)

    for ax, full_dyn, title, cmap in cases:
        stat_pts = sw_static[idx]
        sw_dyn = full_dyn[idx]
        diff = sw_dyn - stat_pts
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        r2 = 1.0 - (np.sum(diff ** 2) / np.sum((stat_pts - np.mean(stat_pts)) ** 2))

        hb = ax.hexbin(stat_pts, sw_dyn, gridsize=40, cmap=cmap, mincnt=1, bins='log')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='1:1 Perfect Match')

        ax.set_title(f"{title}\nRMSE = {rmse:.4f} | MAE = {mae:.4f} | R^2 = {r2:.4f}", fontweight='bold')
        ax.set_xlabel("Static Water Saturation Sw_static [frac]")
        ax.set_ylabel("Dynamic Water Saturation Sw_dynamic [frac]")
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.02])
        ax.legend(loc='upper left', frameon=True)
        fig.colorbar(hb, ax=ax, label='Log Cell Count')

    plt.suptitle("Static vs. Dynamic Saturation Parity Cross-Plots", fontweight='bold')
    plt.tight_layout()
    return fig


def plot_error_distributions(
    sw_static: np.ndarray,
    sw_dynamic_init: np.ndarray,
    sw_dynamic_opt: np.ndarray,
    sw_dynamic_mult: Optional[np.ndarray] = None,
    rt_active: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Histogram and Boxplot of saturation residuals (Sw_dyn - Sw_stat) showing variance collapse.
    """
    set_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    err_init = sw_dynamic_init - sw_static
    err_opt = sw_dynamic_opt - sw_static

    # 1. Global Error Histogram
    ax0 = axes[0]
    sns.kdeplot(err_init, ax=ax0, color='blue', fill=True, alpha=0.25, label=f'Baseline (Std={np.std(err_init):.4f})')
    if sw_dynamic_mult is not None:
        err_mult = sw_dynamic_mult - sw_static
        sns.kdeplot(err_mult, ax=ax0, color='purple', fill=True, alpha=0.3, label=f'Multiplier-Only (Std={np.std(err_mult):.4f})')
    sns.kdeplot(err_opt, ax=ax0, color='red', fill=True, alpha=0.4, label=f'Multiplier+Endpoint (Std={np.std(err_opt):.4f})')
    ax0.axvline(0.0, color='black', linestyle='--', lw=1.5)
    ax0.set_title("Global Residual Error Distribution (Sw_dyn - Sw_stat)", fontweight='bold')
    ax0.set_xlabel("Saturation Residual Delta Sw [frac]")
    ax0.set_ylabel("Density")
    ax0.legend(frameon=True)

    # 2. Per-Rock-Type Residual Boxplots
    ax1 = axes[1]
    if rt_active is not None:
        df_box = []
        for rt in [1, 2, 3]:
            mask = (rt_active == rt)
            if np.any(mask):
                sample_k = min(1000, int(np.sum(mask)))
                for e in np.random.choice(err_init[mask], sample_k, replace=False):
                    df_box.append({'Rock Type': f'RT {rt}', 'Error': e, 'Stage': '1. Baseline'})
                if sw_dynamic_mult is not None:
                    err_m = sw_dynamic_mult - sw_static
                    for e in np.random.choice(err_m[mask], sample_k, replace=False):
                        df_box.append({'Rock Type': f'RT {rt}', 'Error': e, 'Stage': '2. Multiplier-Only'})
                for e in np.random.choice(err_opt[mask], sample_k, replace=False):
                    df_box.append({'Rock Type': f'RT {rt}', 'Error': e, 'Stage': '3. Multiplier+Endpoint'})

        df_plot = pd.DataFrame(df_box)
        palette = ['lightblue', 'plum', 'lightcoral'] if sw_dynamic_mult is not None else ['lightblue', 'lightcoral']
        sns.boxplot(data=df_plot, x='Rock Type', y='Error', hue='Stage', ax=ax1, palette=palette)
        ax1.axhline(0.0, color='black', linestyle='--', lw=1.2)
        ax1.set_title("Per-Rock-Type Error Variance Collapse Across Stages", fontweight='bold')
        ax1.set_ylabel("Saturation Residual Delta Sw [frac]")
        ax1.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    return fig
