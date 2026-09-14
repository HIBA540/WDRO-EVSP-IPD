"""Point d'entrée : lance le pipeline complet WDRO-EVSP v10."""
import os
import sys

# Assure que la racine du projet est dans le PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.main import main_v10_full


if __name__ == '__main__':
    results = main_v10_full(
        data_path_root='data',
        output_root='output_v10_full',
    )
    print("\n[Done] Résultats:", results['out_root'])