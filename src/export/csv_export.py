"""Export CSV."""
import os
import numpy as np
import pandas as pd

from config.settings import CALIBRATION_METHODS


def export_csv(all_res, calib_results, out_dir):
    rows = []
    for cname, cr in all_res.items():
        for mk, mr in cr.items():
            rows.append({
                'case': cname,
                'method': mk,
                'label': CALIBRATION_METHODS.get(mk, {}).get('label', mk),
                'family': CALIBRATION_METHODS.get(mk, {}).get('family', ''),
                'rho': calib_results.get(mk, {}).get('rho', np.nan),
                'E_OOS': mr.get('E_OOS', np.nan),
                'CVaR90': mr.get('CVaR90', np.nan),
                'reliability': mr.get('reliability', np.nan),
                'J_IS': mr.get('J_IS', np.nan),
                'delta_J_pct': mr.get('delta_J_pct', np.nan),
                'composite_score': mr.get('composite_score', np.nan),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'all_results.csv')
    df.to_csv(path, index=False)
    print(f"  [CSV] Saved: {path}")
    return df