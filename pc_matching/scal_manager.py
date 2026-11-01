"""
pc_matching.scal_manager
========================
Manager for Special Core Analysis (SCAL) saturation functions.
Parses, modifies, validates, and exports ECLIPSE/tNavigator SWOF and SGOF tables.
Supports vertical Pc multiplier scaling, Swirr / Pce endpoint adjustments,
and power-law curvature exponent (beta) shape tuning.
"""

import os
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class SCALTable:
    """
    Data container for a single Rock Type SCAL definition.
    """
    def __init__(
        self,
        rt_id: int,
        swof_df: pd.DataFrame,
        sgof_df: Optional[pd.DataFrame] = None,
        fileunit: str = "METRIC",
        source_path: Optional[str] = None
    ):
        self.rt_id = rt_id
        self.swof_df = swof_df.copy()
        self.base_swof_df = swof_df.copy()
        self.sgof_df = sgof_df.copy() if sgof_df is not None else None
        self.fileunit = fileunit
        self.source_path = source_path
        self.current_multiplier = 1.0
        self.current_swirr = float(self.base_swof_df['Sw'].iloc[0])
        self.current_pce = 0.0
        self.current_beta = 1.0

    @property
    def sw(self) -> np.ndarray:
        return self.swof_df['Sw'].values

    @property
    def base_sw(self) -> np.ndarray:
        return self.base_swof_df['Sw'].values

    @property
    def krw(self) -> np.ndarray:
        return self.swof_df['Krw'].values

    @property
    def base_krw(self) -> np.ndarray:
        return self.base_swof_df['Krw'].values

    @property
    def krow(self) -> np.ndarray:
        return self.swof_df['Krow'].values

    @property
    def base_krow(self) -> np.ndarray:
        return self.base_swof_df['Krow'].values

    @property
    def pcow(self) -> np.ndarray:
        return self.swof_df['Pcow'].values

    @property
    def base_pcow(self) -> np.ndarray:
        return self.base_swof_df['Pcow'].values

    @property
    def base_swirr(self) -> float:
        return float(self.base_swof_df['Sw'].iloc[0])

    @property
    def sw_irr(self) -> float:
        return float(self.swof_df['Sw'].iloc[0])

    @property
    def pc_max(self) -> float:
        return float(self.swof_df['Pcow'].iloc[0])


class SCALManager:
    """
    Manages multi-rock-type SCAL models and generates simulation input decks.
    Supports Pc multiplier scaling, Swirr / Pce endpoint adjustments, and curvature beta tuning.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.tables: Dict[int, SCALTable] = {}

    def load_rock_type(self, rt_id: int, filename: Optional[str] = None) -> SCALTable:
        """
        Load SWOF and SGOF from a rock-type GRDECL file.
        """
        if filename is None:
            filename = f"RT-{rt_id}.GRDECL"
        filepath = filename if os.path.isabs(filename) else os.path.join(self.workspace_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SCAL file not found: {filepath}")

        swof_rows = []
        sgof_rows = []
        fileunit = "METRIC"
        current_section = None

        with open(filepath, 'r') as f:
            for line in f:
                clean = line.strip()
                if not clean or clean.startswith('--'):
                    continue
                if clean.startswith('SWOF'):
                    current_section = 'SWOF'
                    continue
                elif clean.startswith('SGOF'):
                    current_section = 'SGOF'
                    continue
                elif clean.startswith('FILEUNIT'):
                    current_section = 'FILEUNIT'
                    continue

                if clean.startswith('/'):
                    current_section = None
                    continue

                if current_section == 'SWOF':
                    parts = [float(x) for x in clean.split()]
                    if len(parts) >= 4:
                        swof_rows.append(parts[:4])
                elif current_section == 'SGOF':
                    parts = [float(x) for x in clean.split()]
                    if len(parts) >= 4:
                        sgof_rows.append(parts[:4])
                elif current_section == 'FILEUNIT':
                    parts = clean.split()
                    if parts:
                        fileunit = parts[0].replace('/', '').strip()

        swof_df = pd.DataFrame(swof_rows, columns=['Sw', 'Krw', 'Krow', 'Pcow'])
        sgof_df = pd.DataFrame(sgof_rows, columns=['Sg', 'Krg', 'Krog', 'Pcog']) if sgof_rows else None

        table = SCALTable(rt_id, swof_df, sgof_df, fileunit, filepath)
        self.tables[rt_id] = table
        return table

    def load_all_rock_types(self, rt_ids: List[int] = [1, 2, 3, 4]) -> Dict[int, SCALTable]:
        """
        Load all rock-type SCAL files.
        """
        for r in rt_ids:
            self.load_rock_type(r)
        return self.tables

    def apply_multiplier(self, rt_id: int, multiplier: float) -> pd.DataFrame:
        """
        Apply simple Pc multiplier to a specific rock type.
        """
        return self.apply_curve_parameters(rt_id, alpha=multiplier)

    def apply_curve_parameters(
        self,
        rt_id: int,
        alpha: float = 1.0,
        swirr: Optional[float] = None,
        pce: float = 0.0,
        beta: float = 1.0
    ) -> pd.DataFrame:
        """
        Adjust capillary pressure curve with:
        1. Vertical Pc multiplier: alpha (> 0)
        2. Connate water saturation endpoint: swirr in (0, 1)
        3. Entry pressure endpoint: pce (>= 0)
        4. Curvature / pore-size distribution shape exponent: beta (> 0)
        
        Preserves relative permeability shapes and guarantees physical monotonicity:
        - Pc(Sw=1.0) = 0.0
        - Pc(Sw) >= 0.0
        - dPc / dSw <= 0
        """
        if rt_id not in self.tables:
            raise KeyError(f"Rock type {rt_id} not loaded in SCALManager.")

        table = self.tables[rt_id]
        if alpha <= 0.0:
            raise ValueError(f"Pc multiplier alpha must be > 0, got {alpha}")

        beta_val = float(beta) if beta > 0.0 else 1.0
        table.current_multiplier = float(alpha)
        table.current_beta = beta_val
        table.current_pce = float(max(0.0, pce))

        base_sw = table.base_sw
        base_swirr = table.base_swirr
        base_pc = table.base_pcow

        s_wn = np.clip((base_sw - base_swirr) / (1.0 - base_swirr), 0.0, 1.0)
        s_wn_curv = s_wn ** beta_val

        # Saturation endpoint adjustment
        if swirr is not None and 0.0 < swirr < 1.0:
            table.current_swirr = float(swirr)
            new_sw = swirr + s_wn_curv * (1.0 - swirr)
        else:
            table.current_swirr = base_swirr
            new_sw = base_swirr + s_wn_curv * (1.0 - base_swirr)

        # Pc scaling with entry pressure
        scaled_pc = base_pc * alpha + table.current_pce * (1.0 - s_wn_curv)

        # Enforce anchoring at Sw=1.0 and non-negativity
        scaled_pc[-1] = 0.0
        scaled_pc = np.maximum(0.0, scaled_pc)

        # Enforce non-increasing monotonicity (from Sw=1 towards Sw_irr)
        for i in range(len(scaled_pc) - 2, -1, -1):
            if scaled_pc[i] < scaled_pc[i + 1]:
                scaled_pc[i] = scaled_pc[i + 1]

        table.swof_df['Sw'] = new_sw
        table.swof_df['Pcow'] = scaled_pc
        return table.swof_df

    def apply_multipliers_dict(self, mult_dict: Dict[int, Union[float, Dict[str, float]]]) -> None:
        """
        Apply parameters to multiple rock types simultaneously.
        """
        for rt_id, val in mult_dict.items():
            if rt_id not in self.tables:
                continue
            if isinstance(val, (int, float)):
                self.apply_curve_parameters(rt_id, alpha=float(val))
            elif isinstance(val, dict):
                self.apply_curve_parameters(
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
                self.apply_curve_parameters(rt_id, alpha=alpha, swirr=swirr, pce=pce, beta=beta)

    def export_scal_file(
        self,
        rt_id: int,
        output_filename: Optional[str] = None,
        backup_original: bool = True,
        stage_label: Optional[str] = None
    ) -> str:
        """
        Export modified SCAL table into a simulation-ready ECLIPSE/tNavigator GRDECL file.
        """
        if rt_id not in self.tables:
            raise KeyError(f"Rock type {rt_id} not loaded.")

        table = self.tables[rt_id]
        if output_filename is None:
            output_filename = f"RT-{rt_id}.GRDECL"

        out_path = os.path.join(self.workspace_dir, output_filename)

        # Backup original if overwriting
        if backup_original and os.path.exists(out_path) and out_path == table.source_path:
            backup_path = out_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copyfile(out_path, backup_path)

        with open(out_path, 'w') as f:
            f.write("SWOF                                   -- Generated : Pc-Matching Optimizer\n")
            if stage_label:
                f.write(f"-- Optimization Formulation : {stage_label}\n")
            f.write(f"-- Modified with Multiplier alpha={table.current_multiplier:.4f}, "
                    f"Sw_irr={table.current_swirr:.4f}, Pce={table.current_pce:.4f}, Beta={table.current_beta:.4f}\n")
            for _, row in table.swof_df.iterrows():
                f.write(f"  {row['Sw']:18.10g}  {row['Krw']:18.10g}  {row['Krow']:18.10g}  {row['Pcow']:18.10g}\n")
            f.write("  /\n\n")

            if table.sgof_df is not None:
                f.write("SGOF                                   -- Generated : Pc-Matching Optimizer\n")
                for _, row in table.sgof_df.iterrows():
                    f.write(f"  {row['Sg']:18.10g}  {row['Krg']:18.10g}  {row['Krog']:18.10g}  {row['Pcog']:18.10g}\n")
                f.write("  /\n\n")

            f.write("FILEUNIT                               -- Generated : Pc-Matching Optimizer\n")
            f.write(f"  {table.fileunit} /\n")

        return out_path

    def export_scal_string(self, rt_id: int, stage_label: Optional[str] = None) -> str:
        """
        Export modified SCAL table as a formatted GRDECL string.
        """
        if rt_id not in self.tables:
            raise KeyError(f"Rock type {rt_id} not loaded.")

        table = self.tables[rt_id]
        lines = []
        lines.append("SWOF                                   -- Generated : Pc-Matching Optimizer\n")
        if stage_label:
            lines.append(f"-- Optimization Formulation : {stage_label}\n")
        lines.append(f"-- Modified with Multiplier alpha={table.current_multiplier:.4f}, "
                     f"Sw_irr={table.current_swirr:.4f}, Pce={table.current_pce:.4f}, Beta={table.current_beta:.4f}\n")
        for _, row in table.swof_df.iterrows():
            lines.append(f"  {row['Sw']:18.10g}  {row['Krw']:18.10g}  {row['Krow']:18.10g}  {row['Pcow']:18.10g}\n")
        lines.append("  /\n\n")

        if table.sgof_df is not None:
            lines.append("SGOF                                   -- Generated : Pc-Matching Optimizer\n")
            for _, row in table.sgof_df.iterrows():
                lines.append(f"  {row['Sg']:18.10g}  {row['Krg']:18.10g}  {row['Krog']:18.10g}  {row['Pcog']:18.10g}\n")
            lines.append("  /\n\n")

        lines.append("FILEUNIT                               -- Generated : Pc-Matching Optimizer\n")
        lines.append(f"  {table.fileunit} /\n")
        return "".join(lines)

    def load_from_props_file(self, filepath: str) -> Dict[int, SCALTable]:
        """
        Parse all SWOF and SGOF tables from an ECLIPSE PROPS include file.
        Assigns rock type IDs incrementally (1, 2, 3, ...).
        """
        with open(filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        swof_tables = []
        sgof_tables = []
        current_rows = []
        in_swof = False
        in_sgof = False
        fileunit = "METRIC"

        for line in lines:
            clean = line.split('--')[0].strip()
            if not clean:
                continue
            if clean.startswith('FILEUNIT'):
                parts = clean.split()
                if len(parts) > 1:
                    fileunit = parts[1].replace('/', '').strip()
                continue
            if clean.startswith('SWOF'):
                in_swof = True
                in_sgof = False
                current_rows = []
                continue
            elif clean.startswith('SGOF'):
                in_sgof = True
                in_swof = False
                current_rows = []
                continue

            if in_swof:
                if '/' in clean:
                    before_slash = clean.split('/')[0].strip()
                    if before_slash:
                        parts = [float(x) for x in before_slash.split()]
                        if len(parts) >= 4:
                            current_rows.append(parts[:4])
                    if current_rows:
                        df = pd.DataFrame(current_rows, columns=['Sw', 'Krw', 'Krow', 'Pcow'])
                        swof_tables.append(df)
                        current_rows = []
                    continue
                else:
                    parts = [float(x) for x in clean.split()]
                    if len(parts) >= 4:
                        current_rows.append(parts[:4])

            elif in_sgof:
                if '/' in clean:
                    before_slash = clean.split('/')[0].strip()
                    if before_slash:
                        parts = [float(x) for x in before_slash.split()]
                        if len(parts) >= 4:
                            current_rows.append(parts[:4])
                    if current_rows:
                        df = pd.DataFrame(current_rows, columns=['Sg', 'Krg', 'Krog', 'Pcog'])
                        sgof_tables.append(df)
                        current_rows = []
                    continue
                else:
                    parts = [float(x) for x in clean.split()]
                    if len(parts) >= 4:
                        current_rows.append(parts[:4])

        loaded = {}
        for idx, swof_df in enumerate(swof_tables, start=1):
            sgof_df = sgof_tables[idx - 1] if idx - 1 < len(sgof_tables) else None
            table = SCALTable(rt_id=idx, swof_df=swof_df, sgof_df=sgof_df, fileunit=fileunit, source_path=filepath)
            self.tables[idx] = table
            loaded[idx] = table
        return loaded

    def export_all(
        self,
        params_dict: Dict[int, Any],
        prefix: str = "opt",
        backup_original: bool = True
    ) -> Dict[int, str]:
        """
        Batch export updated SCAL files for all rock types.
        """
        exported_files = {}
        for rt_id, val in params_dict.items():
            if rt_id in self.tables:
                if isinstance(val, (int, float)):
                    self.apply_curve_parameters(rt_id, alpha=float(val))
                elif isinstance(val, dict):
                    self.apply_curve_parameters(
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
                    self.apply_curve_parameters(rt_id, alpha=alpha, swirr=swirr, pce=pce, beta=beta)

                fname = f"RT-{rt_id}_{prefix}.GRDECL" if prefix else f"RT-{rt_id}.GRDECL"
                exported_files[rt_id] = self.export_scal_file(rt_id, fname, backup_original=backup_original)
        return exported_files
