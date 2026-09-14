"""Génération de toutes les figures pour l'article."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})
import matplotlib.pyplot as plt

from config.settings import (
    CALIBRATION_METHODS, ALL_CASES, XI_FEATURE_NAMES, D_EFF,
)


def make_all_figures(calib_results, all_res, all_op_kpis, Xi_IS,
                      forecast_quality, out_dir):
    # Fig 1 : temps de calibration
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, times = [], []
    for mk, mv in calib_results.items():
        labels.append(CALIBRATION_METHODS.get(mk, {}).get('label', mk))
        times.append(mv.get('calib_time', 0.) * 1000)
    colors = [CALIBRATION_METHODS.get(mk, {}).get('color', 'gray') for mk in calib_results]
    ax.bar(labels, times, color=colors)
    ax.set_ylabel('Calibration time (ms)')
    ax.set_yscale('log')
    ax.set_title('Calibration time per method')
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_calibration_time.png'), dpi=200)
    plt.close(fig)

    # Fig 2 : rayon par méthode
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mk in enumerate(calib_results):
        ax.bar(i, calib_results[mk]['rho'],
               color=CALIBRATION_METHODS.get(mk, {}).get('color', 'gray'),
               label=CALIBRATION_METHODS.get(mk, {}).get('label', mk))
    ax.set_xticks(range(len(calib_results)))
    ax.set_xticklabels([CALIBRATION_METHODS.get(mk, {}).get('label', mk)
                        for mk in calib_results], rotation=15)
    ax.set_ylabel('Calibrated radius ρ')
    ax.set_title('Calibrated radius per method')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_radius_per_method.png'), dpi=200)
    plt.close(fig)

    # Fig 3 : E_OOS par régime
    regimes = sorted(set(ALL_CASES[c].get('regime', 'unknown') for c in ALL_CASES))
    methods = list(calib_results.keys())
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(methods)
    x = np.arange(len(regimes))
    for i, mk in enumerate(methods):
        vals = []
        for reg in regimes:
            v = [all_res[c][mk].get('E_OOS', np.nan) for c in ALL_CASES
                 if ALL_CASES[c].get('regime') == reg and mk in all_res.get(c, {})]
            vals.append(np.nanmean(v) if v else np.nan)
        ax.bar(x + i * width, vals, width,
               label=CALIBRATION_METHODS.get(mk, {}).get('label', mk),
               color=CALIBRATION_METHODS.get(mk, {}).get('color', 'gray'))
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(regimes, rotation=15)
    ax.set_ylabel('E_OOS (€)')
    ax.set_title('E_OOS by regime')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_eoos_regime.png'), dpi=200)
    plt.close(fig)

    # Fig 4 : CVaR90 scénarios sévères
    severe = ['C8-Storm', 'S4-HeavyTail', 'S5-Adversarial']
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(severe))
    width = 0.8 / len(methods)
    for i, mk in enumerate(methods):
        vals = [all_res.get(c, {}).get(mk, {}).get('CVaR90', np.nan) for c in severe]
        ax.bar(x + i * width, vals, width,
               label=CALIBRATION_METHODS.get(mk, {}).get('label', mk),
               color=CALIBRATION_METHODS.get(mk, {}).get('color', 'gray'))
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(severe)
    ax.set_ylabel('CVaR90 (€)')
    ax.set_title('CVaR90 in severe scenarios')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_cvar_severe.png'), dpi=200)
    plt.close(fig)

    # Fig 5 : corrélation ξ
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = np.corrcoef(Xi_IS.T)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(D_EFF)); ax.set_xticklabels(XI_FEATURE_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(D_EFF)); ax.set_yticklabels(XI_FEATURE_NAMES, fontsize=7)
    ax.set_title('Correlation matrix of ξ')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_correlation_xi.png'), dpi=200)
    plt.close(fig)

    # Fig 6 : qualité de prévision
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(['RMSE', 'Coverage P5-P95'],
           [forecast_quality.get('rmse', np.nan),
            forecast_quality.get('coverage', np.nan)],
           color=['#3498DB', '#C0392B'])
    ax.set_title('Forecast quality')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_forecast_quality.png'), dpi=200)
    plt.close(fig)

    # Fig 7 : décomposition par bloc
    if 'rho_fics' in calib_results and 'block_diagnostics' in calib_results['rho_fics']:
        bd = calib_results['rho_fics']['block_diagnostics']
        blocks = list(bd.keys())
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(blocks))
        w = 0.35
        retro = [bd[b]['R_retro'] for b in blocks]
        prosp = [bd[b]['R_prosp'] for b in blocks]
        ax.bar(x - w / 2, retro, w, label='Retrospective', color='#3498DB')
        ax.bar(x + w / 2, prosp, w, label='Prospective', color='#C0392B')
        ax.set_xticks(x); ax.set_xticklabels(blocks)
        ax.set_ylabel('Radius contribution')
        ax.set_title('IPD block decomposition')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_block_decomposition.png'), dpi=200)
        plt.close(fig)

    print(f"  [Figures] Saved: {out_dir}/fig_*.png")