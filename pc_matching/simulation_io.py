"""
pc_matching.simulation_io
=========================
Manages manual tNavigator execution integration:
- Generates modified simulation include files (RT-*.GRDECL).
- Supports both multiplier and endpoint adjusted parameter sets.
- Imports dynamic initial Sw properties exported from tNavigator.
- Logs and tracks human-in-the-loop manual optimization trials.
"""

import os
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from .scal_manager import SCALManager
from .grdecl_io import FastGRDECLParser


class TNavWorkflowManager:
    """
    Coordinates simulation deck generation and results ingestion for manual tNavigator runs.
    """

    def __init__(self, workspace_dir: str, scal_manager: SCALManager):
        self.workspace_dir = workspace_dir
        self.scal_manager = scal_manager
        self.history_records: List[Dict[str, Any]] = []

    def export_iteration_decks(
        self,
        iteration: int,
        multipliers_dict: Dict[int, Any],
        backup: bool = True
    ) -> Dict[str, str]:
        """
        Generate modified SCAL files for tNavigator manual run.
        Supports scalar multipliers or endpoint parameter dictionaries.
        Can overwrite RT-*.GRDECL (with backup) or write versioned RT-*_iterN.GRDECL.
        """
        exported = {}
        for rt_id, val in multipliers_dict.items():
            if isinstance(val, (int, float)):
                self.scal_manager.apply_curve_parameters(rt_id, alpha=float(val))
            elif isinstance(val, dict):
                self.scal_manager.apply_curve_parameters(
                    rt_id,
                    alpha=float(val.get('alpha', 1.0)),
                    swirr=val.get('swirr', None),
                    pce=float(val.get('pce', 0.0)),
                    beta=float(val.get('beta', 1.0))
                )
            elif isinstance(val, (tuple, list)):
                alpha = float(val[0])
                swirr = float(val[1]) if len(val) > 1 else None
                pce = float(val[2]) if len(val) > 2 else 0.0
                beta = float(val[3]) if len(val) > 3 else 1.0
                self.scal_manager.apply_curve_parameters(rt_id, alpha=alpha, swirr=swirr, pce=pce, beta=beta)

            # Standard simulation filename
            std_filename = f"RT-{rt_id}.GRDECL"
            std_path = os.path.join(self.workspace_dir, std_filename)

            if backup and os.path.exists(std_path):
                backup_path = os.path.join(self.workspace_dir, f"RT-{rt_id}_backup.GRDECL")
                if not os.path.exists(backup_path):
                    shutil.copyfile(std_path, backup_path)

            # Export standard file for direct tNav deck inclusion
            out_path = self.scal_manager.export_scal_file(rt_id, std_filename, backup_original=False)
            exported[f"RT-{rt_id}"] = out_path

            # Also save versioned copy for reproducibility
            iter_filename = f"RT-{rt_id}_iter{iteration}.GRDECL"
            self.scal_manager.export_scal_file(rt_id, iter_filename, backup_original=False)

        return exported

    def import_dynamic_sw(
        self,
        filename: str = "SWAT_DYNAMIC.GRDECL",
        keyword: str = "SWAT",
        total_cells: Optional[int] = None
    ) -> np.ndarray:
        """
        Import dynamic Sw 3D grid property exported from tNavigator after equilibration.
        """
        filepath = os.path.join(self.workspace_dir, filename) if not os.path.isabs(filename) else filename
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"tNavigator dynamic Sw file not found at: {filepath}\n"
                f"Please run initialization in tNavigator and export the SWAT property to '{filename}'."
            )

        sw_dynamic_grid = FastGRDECLParser.parse_property(filepath, keyword, total_cells=total_cells)
        return sw_dynamic_grid

    def record_trial(
        self,
        iteration: int,
        multipliers_dict: Dict[int, Any],
        metrics: Dict[str, float],
        notes: str = ""
    ) -> pd.DataFrame:
        """
        Record a trial iteration in the tracking log.
        """
        param_cols = {}
        for k, v in multipliers_dict.items():
            if isinstance(v, dict):
                param_cols[f'Alpha_RT{k}'] = v.get('alpha', 1.0)
                if 'swirr' in v and v['swirr'] is not None:
                    param_cols[f'Swirr_RT{k}'] = v['swirr']
                if 'pce' in v and v['pce'] is not None:
                    param_cols[f'Pce_RT{k}'] = v['pce']
                if 'beta' in v and v['beta'] is not None:
                    param_cols[f'Beta_RT{k}'] = v['beta']
            elif isinstance(v, (tuple, list)):
                param_cols[f'Alpha_RT{k}'] = v[0]
                if len(v) > 1 and v[1] is not None:
                    param_cols[f'Swirr_RT{k}'] = v[1]
                if len(v) > 2 and v[2] is not None:
                    param_cols[f'Pce_RT{k}'] = v[2]
                if len(v) > 3 and v[3] is not None:
                    param_cols[f'Beta_RT{k}'] = v[3]
            else:
                param_cols[f'Alpha_RT{k}'] = v

        record = {
            'Iteration': iteration,
            **param_cols,
            'Global_RMSE': metrics.get('global_rmse', np.nan),
            'Global_MAE': metrics.get('global_mae', np.nan),
            'R2': metrics.get('r2_score', np.nan),
            'Notes': notes
        }
        self.history_records.append(record)
        return pd.DataFrame(self.history_records)

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Return audit history as a DataFrame.
        """
        return pd.DataFrame(self.history_records)
