"""Export JSON."""
import json
import datetime
import numpy as np


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def export_json(calib_results, all_res, all_op_kpis, forecast_quality,
                d_eff, stationarity_results, out_path):
    payload = {
        'timestamp': datetime.datetime.now().isoformat(),
        'calibration': _to_serializable(calib_results),
        'oos_results': _to_serializable(all_res),
        'operational_kpis': _to_serializable(all_op_kpis),
        'forecast_quality': _to_serializable(forecast_quality),
        'intrinsic_dimension': float(d_eff),
        'stationarity': _to_serializable(stationarity_results),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [JSON] Saved: {out_path}")