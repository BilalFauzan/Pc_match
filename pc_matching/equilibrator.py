"""
pc_matching.equilibrator
========================
Vectorized 3D Reservoir Equilibration Emulator.
Simulates analytical tNavigator / ECLIPSE hydrostatic initialization
from Free Water Level (FWL), fluid gradients, and capillary pressure curves.
Supports multiplier scaling, Swirr / Pce endpoint adjustments, and curvature beta tuning.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, Union
from .scal_manager import SCALManager


class EquilibrationParams:
    """
    Equilibration and fluid parameters matching ECLIPSE / tNavigator EQUIL keyword.
    Supports both METRIC (m, bar, kg/m^3) and FIELD (ft, psi, lb/ft^3) units.
    """
    def __init__(
        self,
        fwl_depth: float = 925.0,        # Free Water Level / WOC TVD [m or ft]
        datum_depth: float = 700.0,      # Datum depth [m or ft]
        datum_pressure: float = 100.0,   # Datum pressure [bar or psi]
        water_density: float = 1020.0,   # Brine density [kg/m^3 or lb/ft^3]
        oil_density: float = 820.0,      # Reservoir oil density [kg/m^3 or lb/ft^3]
        unit_system: str = "METRIC",      # Unit convention: 'METRIC' or 'FIELD'
        g: float = 9.80665               # Acceleration of gravity [m/s^2]
    ):
        self.fwl_depth = fwl_depth
        self.datum_depth = datum_depth
        self.datum_pressure = datum_pressure
        self.water_density = water_density
        self.oil_density = oil_density
        self.unit_system = unit_system.upper().strip()
        self.g = g

    @property
    def density_difference(self) -> float:
        """Delta rho in kg/m^3 or lb/ft^3."""
        return max(0.0, self.water_density - self.oil_density)

    @property
    def capillary_gradient(self) -> float:
        """
        Capillary pressure gradient:
        - METRIC: (rho_w - rho_o) [kg/m^3] * g * 1e-5  --> [bar/m]
        - FIELD:  (rho_w - rho_o) [lb/ft^3] / 144      --> [psi/ft]
        """
        if self.unit_system == "FIELD":
            return self.density_difference / 144.0
        else:
            return self.density_difference * self.g * 1e-5

    @property
    def pressure_unit(self) -> str:
        return "psi" if self.unit_system == "FIELD" else "bar"

    @property
    def depth_unit(self) -> str:
        return "ft" if self.unit_system == "FIELD" else "m"

    @property
    def density_unit(self) -> str:
        return "lb/ft³" if self.unit_system == "FIELD" else "kg/m³"

    @classmethod
    def from_includes(
        cls,
        sol_filepath: str,
        props_filepath: str,
        eql_region: int = 1,
        pvt_region: int = 1,
        unit_system: Optional[str] = None,
        g: float = 9.80665
    ) -> 'EquilibrationParams':
        """
        Load equilibration and density parameters directly from ECLIPSE include files.
        Auto-detects unit convention (METRIC vs FIELD) if not specified.
        """
        from .grdecl_io import FastGRDECLParser
        equil_list = FastGRDECLParser.parse_equil_keyword(sol_filepath)
        density_list = FastGRDECLParser.parse_density_keyword(props_filepath)

        if not equil_list:
            raise ValueError(f"No EQUIL records found in {sol_filepath}")
        if not density_list:
            raise ValueError(f"No DENSITY records found in {props_filepath}")

        eq_rec = equil_list[eql_region - 1] if eql_region <= len(equil_list) else equil_list[0]
        dens_idx = (pvt_region - 1) if pvt_region <= len(density_list) else (1 if len(density_list) > 1 else 0)
        dens_rec = density_list[dens_idx]

        w_dens = float(dens_rec['water_density'])
        o_dens = float(dens_rec['oil_density'])

        # Auto-detect unit system from density value if not specified
        if unit_system is None:
            unit_system = "FIELD" if w_dens < 200.0 else "METRIC"

        return cls(
            fwl_depth=float(eq_rec['woc_depth']),
            datum_depth=float(eq_rec['datum_depth']),
            datum_pressure=float(eq_rec['datum_pressure']),
            water_density=w_dens,
            oil_density=o_dens,
            unit_system=unit_system,
            g=g
        )

        eql_idx = min(max(1, eql_region), len(equil_list)) - 1
        pvt_idx = min(max(1, pvt_region), len(density_list)) - 1

        eq_rec = equil_list[eql_idx]
        dens_rec = density_list[pvt_idx]

        return cls(
            fwl_depth=float(eq_rec['woc_depth']),
            datum_depth=float(eq_rec['datum_depth']),
            datum_pressure=float(eq_rec['datum_pressure']),
            water_density=float(dens_rec['water_density']),
            oil_density=float(dens_rec['oil_density']),
            g=g
        )


class ReservoirEquilibrator:
    """
    Simulates dynamic Sw equilibration across 3D reservoir grid cells.
    """

    def __init__(
        self,
        scal_manager: SCALManager,
        params: Optional[EquilibrationParams] = None,
        multi_params: Optional[Dict[int, EquilibrationParams]] = None
    ):
        self.scal_manager = scal_manager
        self.params = params if params is not None else EquilibrationParams()
        self.multi_params = multi_params

    def compute_hydrostatic_pc(
        self,
        depth_array: np.ndarray,
        eqlnum_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute hydrostatic capillary pressure at cell depth:
        Supports single-region or multi-region (EQLNUM) equilibration.
        """
        if eqlnum_array is not None and self.multi_params is not None:
            pc_hydro = np.zeros_like(depth_array, dtype=np.float32)
            for r_id, p in self.multi_params.items():
                m = (eqlnum_array == r_id)
                if np.any(m):
                    h = np.maximum(0.0, p.fwl_depth - depth_array[m])
                    pc_hydro[m] = h * p.capillary_gradient
            return pc_hydro
        else:
            height_above_fwl = np.maximum(0.0, self.params.fwl_depth - depth_array)
            return height_above_fwl * self.params.capillary_gradient

    def calculate_dynamic_sw(
        self,
        depth_array: np.ndarray,
        rt_array: np.ndarray,
        params_dict: Optional[Dict[int, Any]] = None,
        eqlnum_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate dynamic initial water saturation for an array of active cells.
        
        Parameters
        ----------
        depth_array : np.ndarray
            Array of cell depths (TVD in meters).
        rt_array : np.ndarray
            Array of rock type IDs corresponding to each cell.
        params_dict : dict, optional
            Rock-type-specific parameters. Can be:
            - {rt_id: alpha} (multiplier only)
            - {rt_id: {'alpha': ..., 'swirr': ..., 'pce': ..., 'beta': ...}}
            - {rt_id: (alpha, swirr, pce, beta)}
            Defaults to 1.0.

        Returns
        -------
        np.ndarray
            Array of equilibrated dynamic water saturations Sw_dynamic in [0, 1].
        """
        if params_dict is None:
            params_dict = {}

        pc_hydro = self.compute_hydrostatic_pc(depth_array, eqlnum_array=eqlnum_array)
        sw_dynamic = np.ones_like(depth_array, dtype=np.float32)

        unique_rts = np.unique(rt_array)
        for rt_id in unique_rts:
            if rt_id == 0 or rt_id not in self.scal_manager.tables:
                continue

            cell_mask = (rt_array == rt_id)
            if not np.any(cell_mask):
                continue

            table = self.scal_manager.tables[rt_id]
            param_val = params_dict.get(int(rt_id), 1.0)

            # Extract alpha, swirr, pce, beta
            if isinstance(param_val, (int, float)):
                alpha = float(param_val)
                swirr = None
                pce = 0.0
                beta = 1.0
            elif isinstance(param_val, dict):
                alpha = float(param_val.get('alpha', 1.0))
                swirr = param_val.get('swirr', None)
                pce = float(param_val.get('pce', 0.0))
                beta = float(param_val.get('beta', 1.0))
            elif isinstance(param_val, (tuple, list)):
                alpha = float(param_val[0])
                swirr = float(param_val[1]) if len(param_val) > 1 else None
                pce = float(param_val[2]) if len(param_val) > 2 else 0.0
                beta = float(param_val[3]) if len(param_val) > 3 else 1.0
            else:
                alpha = 1.0
                swirr = None
                pce = 0.0
                beta = 1.0

            base_sw = table.base_sw
            base_swirr = table.base_swirr
            base_pc = table.base_pcow

            # Curvature calculation
            s_wn = np.clip((base_sw - base_swirr) / (1.0 - base_swirr), 0.0, 1.0)
            s_wn_curv = s_wn ** beta

            # Saturation points
            if swirr is not None and 0.0 < swirr < 1.0:
                sw_pts = (swirr + s_wn_curv * (1.0 - swirr))[::-1]
            else:
                sw_pts = (base_swirr + s_wn_curv * (1.0 - base_swirr))[::-1]

            # Scaled Pc points
            scaled_pc = base_pc * alpha + pce * (1.0 - s_wn_curv)
            scaled_pc[-1] = 0.0
            scaled_pc = np.maximum(0.0, scaled_pc)
            for i in range(len(scaled_pc) - 2, -1, -1):
                if scaled_pc[i] < scaled_pc[i + 1]:
                    scaled_pc[i] = scaled_pc[i + 1]
            pc_pts = scaled_pc[::-1]

            # Invert: Pc_hydro -> Sw_dynamic
            sw_dynamic[cell_mask] = np.interp(
                pc_hydro[cell_mask],
                pc_pts,
                sw_pts,
                left=1.0,
                right=sw_pts[-1]
            )

        return sw_dynamic
