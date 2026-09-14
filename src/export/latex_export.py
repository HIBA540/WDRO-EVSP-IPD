"""Export LaTeX."""
import os
import numpy as np
import pandas as pd

from config.settings import ALL_CASES, CALIBRATION_METHODS


def export_latex_tables(all_res, calib_results, out_dir):
    cal_rows = []
    for mk, mv in calib_results.items():
        cal_rows.append({
            'Method': CALIBRATION_METHODS.get(mk, {}).get('label', mk),
            'Family': CALIBRATION_METHODS.get(mk, {}).get('family', ''),
            'rho': f"{mv.get('rho', np.nan):.5f}",
            'Time (ms)': f"{mv.get('calib_time', 0.)*1000:.2f}",
        })
    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_latex(os.path.join(out_dir, 'table_calibration.tex'),
                    index=False, escape=False, caption='Calibrated radii.',
                    label='tab:calibration')

    regimes = sorted(set(ALL_CASES[c].get('regime', 'unknown') for c in ALL_CASES))
    rows = []
    for regime in regimes:
        row = {'Regime': regime}
        for mk in calib_results:
            vals = [all_res[c][mk].get('E_OOS', np.nan) for c in ALL_CASES
                    if ALL_CASES[c].get('regime') == regime and mk in all_res.get(c, {})]
            row[CALIBRATION_METHODS.get(mk, {}).get('label', mk)] = \
                f"{np.nanmean(vals):.1f}" if vals else "nan"
        rows.append(row)
    pd.DataFrame(rows).to_latex(os.path.join(out_dir, 'table_eoos_regime.tex'),
                                 index=False, escape=False,
                                 caption='E\\_OOS by regime.', label='tab:eoos')

    severe = ['C8-Storm', 'S4-HeavyTail', 'S5-Adversarial']
    rows = []
    for cname in severe:
        cr = all_res.get(cname, {})
        row = {'Case': cname}
        for mk in calib_results:
            row[CALIBRATION_METHODS.get(mk, {}).get('label', mk)] = \
                f"{cr.get(mk, {}).get('CVaR90', np.nan):.1f}"
        rows.append(row)
    pd.DataFrame(rows).to_latex(os.path.join(out_dir, 'table_cvar_severe.tex'),
                                 index=False, escape=False,
                                 caption='CVaR90 in severe scenarios.', label='tab:cvar')

    print(f"  [LaTeX] Saved: {out_dir}/table_*.tex")