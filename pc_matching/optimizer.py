"""
pc_matching.optimizer
=====================
Optimization engine for Pc-Sw matching between 3D Static Sw and Dynamic Sw.
Implements spatial mismatch objectives (RMSE, MAE) and rock-type-specific
solvers for:
1. Vertical Pc multiplier optimization (alpha_RT)
2. Joint Pc multiplier and endpoint optimization (alpha_RT, Sw_irr, Pce)
3. Full curvature & endpoint optimization (alpha_RT, Sw_irr, Pce, beta_RT)
Strictly without any hydrocarbon volume / STOIIP dependencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from scipy.optimize import minimize_scalar, minimize, differential_evolution, dual_annealing
from .equilibrator import ReservoirEquilibrator


class OptimizationMetrics:
    """
    Evaluator for spatial saturation matching quality and in-place volumetric consistency.
    """

    @staticmethod
    def compute(
        sw_static: np.ndarray,
        sw_dynamic: np.ndarray,
        rt_array: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive spatial mismatch metrics between static and dynamic Sw,
        including pore-volume weighted metrics and In-Place Hydrocarbon Pore Volume (HCPV) match.
        
        Metrics:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - Max Error
        - Bias (Mean Signed Difference: Sw_dyn - Sw_stat)
        - R2 (Coefficient of Determination)
        - In-Place Match % (HCPV Dynamic vs HCPV Static)
        - Per-Rock-Type RMSE, MAE, and In-Place Match %
        """
        diff = sw_dynamic - sw_static
        sq_diff = diff ** 2
        abs_diff = np.abs(diff)

        rmse = float(np.sqrt(np.mean(sq_diff)))
        mae = float(np.mean(abs_diff))
        max_err = float(np.max(abs_diff))
        bias = float(np.mean(diff))

        # Pearson R^2
        ss_tot = np.sum((sw_static - np.mean(sw_static)) ** 2)
        ss_res = np.sum(sq_diff)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        metrics = {
            'global_rmse': rmse,
            'global_mae': mae,
            'max_abs_error': max_err,
            'mean_bias': bias,
            'r2_score': r2
        }

        # Volumetric / In-Place metrics if weights (Pore Volume) provided
        if weights is not None:
            sum_w = float(np.sum(weights))
            if sum_w > 0:
                weighted_rmse = float(np.sqrt(np.sum(weights * sq_diff) / sum_w))
                weighted_mae = float(np.sum(weights * abs_diff) / sum_w)
                hcpv_stat = float(np.sum(weights * np.maximum(0.0, 1.0 - sw_static)))
                hcpv_dyn = float(np.sum(weights * np.maximum(0.0, 1.0 - sw_dynamic)))
                delta_hcpv = hcpv_dyn - hcpv_stat
                inplace_match_pct = float(max(0.0, (1.0 - abs(delta_hcpv) / hcpv_stat) * 100.0)) if hcpv_stat > 0 else 100.0
                inplace_delta_pct = float((delta_hcpv / hcpv_stat) * 100.0) if hcpv_stat > 0 else 0.0

                metrics.update({
                    'weighted_rmse': weighted_rmse,
                    'weighted_mae': weighted_mae,
                    'hcpv_static': hcpv_stat,
                    'hcpv_dynamic': hcpv_dyn,
                    'inplace_match_pct': inplace_match_pct,
                    'inplace_delta_pct': inplace_delta_pct
                })

        if rt_array is not None:
            unique_rts = np.unique(rt_array)
            for rt in unique_rts:
                if rt == 0:
                    continue
                mask = (rt_array == rt)
                if np.any(mask):
                    metrics[f'rt_{int(rt)}_rmse'] = float(np.sqrt(np.mean(sq_diff[mask])))
                    metrics[f'rt_{int(rt)}_mae'] = float(np.mean(abs_diff[mask]))
                    if weights is not None:
                        w_rt = weights[mask]
                        sum_w_rt = float(np.sum(w_rt))
                        if sum_w_rt > 0:
                            hcpv_s = float(np.sum(w_rt * np.maximum(0.0, 1.0 - sw_static[mask])))
                            hcpv_d = float(np.sum(w_rt * np.maximum(0.0, 1.0 - sw_dynamic[mask])))
                            m_pct = float(max(0.0, (1.0 - abs(hcpv_d - hcpv_s) / hcpv_s) * 100.0)) if hcpv_s > 0 else 100.0
                            metrics[f'rt_{int(rt)}_inplace_match_pct'] = m_pct

        return metrics


def compute_residual_loss(
    err: np.ndarray,
    weights: np.ndarray,
    loss_type: str = "Huber Loss (Robust)",
    delta: float = 0.10
) -> float:
    """
    Compute weighted objective loss supporting Huber, L1 (MAE), or L2 (RMSE).
    Huber loss provides extreme robustness against wide scatter and outliers.
    """
    sum_w = float(np.sum(weights))
    if sum_w <= 0.0:
        return 0.0

    abs_err = np.abs(err)
    lt = loss_type.lower()
    if "huber" in lt:
        huber_err = np.where(abs_err <= delta, 0.5 * (err ** 2), delta * abs_err - 0.5 * (delta ** 2))
        return float(np.sum(weights * huber_err) / sum_w)
    elif "mae" in lt or "l1" in lt:
        return float(np.sum(weights * abs_err) / sum_w)
    else:  # Standard L2 RMSE
        return float(np.sqrt(np.sum(weights * (err ** 2)) / sum_w))


def compute_equilibrium_zone_inplace_metrics(
    eqlnum_array: np.ndarray,
    sw_static: np.ndarray,
    sw_dynamic: np.ndarray,
    poro: np.ndarray,
    depth: np.ndarray,
    ntg: Optional[np.ndarray] = None,
    sw_dynamic_init: Optional[np.ndarray] = None,
    multi_params: Optional[Dict[int, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute reservoir In-Place (HCPV) conformance and Sw saturation matching metrics
    broken down per Equilibrium Zone / Compartment (EQLNUM / EQUIL region).

    Parameters
    ----------
    eqlnum_array : np.ndarray
        Array of equilibrium region numbers (EQLNUM) for each active cell.
    sw_static : np.ndarray
        Static water saturation array (e.g. Petrel SWAT).
    sw_dynamic : np.ndarray
        Optimized dynamic equilibrated water saturation array.
    poro : np.ndarray
        Porosity array.
    depth : np.ndarray
        True vertical depth (TVD) array.
    ntg : np.ndarray, optional
        Net-to-Gross array (defaults to 1.0 if not provided).
    sw_dynamic_init : np.ndarray, optional
        Initial uncalibrated dynamic water saturation array (for before vs after comparison).
    multi_params : dict, optional
        Dictionary mapping region_id -> EquilibrationParams (to extract WOC/FWL and datum).

    Returns
    -------
    List[Dict[str, Any]]
        List of equilibrium zone metric dicts sorted by region ID.
    """
    unique_regions = [int(r) for r in np.unique(eqlnum_array) if r > 0]
    records = []
    w_all = poro * (ntg if ntg is not None else 1.0)

    for r in sorted(unique_regions):
        mask_r = (eqlnum_array == r)
        n_cells = int(np.sum(mask_r))
        if n_cells == 0:
            continue

        z_mean = float(np.mean(depth[mask_r]))
        poro_mean = float(np.mean(poro[mask_r]))
        sw_stat_mean = float(np.mean(sw_static[mask_r]))
        sw_dyn_mean = float(np.mean(sw_dynamic[mask_r]))

        w_r = w_all[mask_r]
        hcpv_stat = float(np.sum(w_r * np.maximum(0.0, 1.0 - sw_static[mask_r])))
        hcpv_dyn = float(np.sum(w_r * np.maximum(0.0, 1.0 - sw_dynamic[mask_r])))

        if hcpv_stat > 1e-6:
            match_pct = float(max(0.0, (1.0 - abs(hcpv_dyn - hcpv_stat) / hcpv_stat) * 100.0))
            delta_pct = float(((hcpv_dyn - hcpv_stat) / hcpv_stat) * 100.0)
        else:
            match_pct = 100.0 if hcpv_dyn < 1e-6 else 0.0
            delta_pct = 0.0 if hcpv_dyn < 1e-6 else 100.0

        rmse = float(np.sqrt(np.mean((sw_dynamic[mask_r] - sw_static[mask_r]) ** 2)))
        mae = float(np.mean(np.abs(sw_dynamic[mask_r] - sw_static[mask_r])))

        woc_val = None
        if multi_params and r in multi_params:
            woc_val = getattr(multi_params[r], 'fwl_depth', None)

        rec = {
            'region': int(r),
            'zone_label': f"Zone {int(r)}" if len(unique_regions) > 1 else "Full Reservoir",
            'woc_depth': woc_val,
            'cells': n_cells,
            'mean_depth': z_mean,
            'mean_poro': poro_mean,
            'mean_sw_stat': sw_stat_mean,
            'mean_sw_dyn': sw_dyn_mean,
            'hcpv_stat': hcpv_stat,
            'hcpv_dyn': hcpv_dyn,
            'inplace_match_pct': match_pct,
            'delta_hcpv_pct': delta_pct,
            'rmse': rmse,
            'mae': mae
        }

        if sw_dynamic_init is not None:
            sw_init_mean = float(np.mean(sw_dynamic_init[mask_r]))
            hcpv_init = float(np.sum(w_r * np.maximum(0.0, 1.0 - sw_dynamic_init[mask_r])))
            if hcpv_stat > 1e-6:
                match_init = float(max(0.0, (1.0 - abs(hcpv_init - hcpv_stat) / hcpv_stat) * 100.0))
            else:
                match_init = 100.0 if hcpv_init < 1e-6 else 0.0
            rmse_init = float(np.sqrt(np.mean((sw_dynamic_init[mask_r] - sw_static[mask_r]) ** 2)))
            rec['mean_sw_init'] = sw_init_mean
            rec['hcpv_init'] = hcpv_init
            rec['inplace_match_init_pct'] = match_init
            rec['rmse_init'] = rmse_init
            rec['match_improvement_pct'] = match_pct - match_init

        records.append(rec)

    return records


# Aliases for backward compatibility
compute_zone_inplace_metrics = compute_equilibrium_zone_inplace_metrics
compute_layer_inplace_metrics = compute_equilibrium_zone_inplace_metrics


class PcOptimizer:
    """
    Automated optimizer for Rock-Type Capillary Pressure multipliers, endpoints, and curvature.
    Supports Pore-Volume Weighting (direct In-Place / STOIIP matching) and Robust Huber Loss.
    """

    def __init__(
        self,
        equilibrator: ReservoirEquilibrator,
        depth_active: np.ndarray,
        rt_active: np.ndarray,
        sw_static_active: np.ndarray,
        eqlnum_active: Optional[np.ndarray] = None,
        poro_active: Optional[np.ndarray] = None,
        ntg_active: Optional[np.ndarray] = None,
        pv_weights: Optional[np.ndarray] = None
    ):
        self.equilibrator = equilibrator
        self.depth_active = depth_active
        self.rt_active = rt_active
        self.sw_static_active = sw_static_active
        self.eqlnum_active = eqlnum_active
        self.poro_active = poro_active
        self.ntg_active = ntg_active

        # Calculate cell pore volume weights for volumetric in-place matching
        if pv_weights is not None:
            self.pv_weights = pv_weights.astype(np.float32)
        elif poro_active is not None:
            ntg = ntg_active if ntg_active is not None else 1.0
            self.pv_weights = np.maximum(1e-4, (poro_active * ntg)).astype(np.float32)
        else:
            self.pv_weights = np.ones_like(depth_active, dtype=np.float32)

        self.active_rts = [int(r) for r in np.unique(rt_active) if r > 0 and r in equilibrator.scal_manager.tables]
        self.iteration_history = []

    def objective_function(
        self,
        params_dict: Dict[int, Any],
        weights: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Evaluate composite spatial mismatch objective:
        J = sum_r (w_r * RMSE_r)
        Strictly based on spatial saturation mismatch (no STOIIP/volume terms).
        """
        sw_dyn = self.equilibrator.calculate_dynamic_sw(
            self.depth_active,
            self.rt_active,
            params_dict,
            eqlnum_array=self.eqlnum_active
        )

        if weights is None:
            # Standard unweighted global RMSE
            return float(np.sqrt(np.mean((sw_dyn - self.sw_static_active) ** 2)))

        total_loss = 0.0
        total_w = 0.0
        for rt in self.active_rts:
            mask = (self.rt_active == rt)
            if np.any(mask):
                w = weights.get(rt, 1.0)
                rt_rmse = np.sqrt(np.mean((sw_dyn[mask] - self.sw_static_active[mask]) ** 2))
                total_loss += w * rt_rmse
                total_w += w

        return float(total_loss / total_w) if total_w > 0 else 0.0

    def optimize_independent(
        self,
        bounds: Tuple[float, float] = (0.01, 50.0),
        tolerance: float = 1e-4,
        weighting_mode: str = "Pore Volume (PV-Weighted)",
        loss_type: str = "Huber Loss (Robust to Scatter)",
        huber_delta: float = 0.10
    ) -> Dict[str, Any]:
        """
        Stage 1: Multiplier-only optimization using 1D bounded Brent search.
        Optimizes alpha_r for each rock type with fixed baseline endpoints.
        Supports Pore Volume Weighting and Huber Loss.
        """
        optimal_multipliers = {}
        rt_results = {}
        use_pv = "pore volume" in weighting_mode.lower() or "pv" in weighting_mode.lower()

        pc_hydro = self.equilibrator.compute_hydrostatic_pc(self.depth_active, eqlnum_array=self.eqlnum_active)

        for rt in self.active_rts:
            mask = (self.rt_active == rt)
            if not np.any(mask):
                continue

            pc_h = pc_hydro[mask]
            target_sw = self.sw_static_active[mask]
            cell_w = self.pv_weights[mask] if use_pv else np.ones(np.sum(mask), dtype=np.float32)
            table = self.equilibrator.scal_manager.tables[rt]

            pc_pts = table.base_pcow[::-1]
            sw_pts = table.base_sw[::-1]

            def rt_loss(alpha: float) -> float:
                pc_eff = pc_h / alpha
                sw_est = np.interp(pc_eff, pc_pts, sw_pts, left=1.0, right=sw_pts[-1])
                return compute_residual_loss(sw_est - target_sw, cell_w, loss_type=loss_type, delta=huber_delta)

            init_loss = rt_loss(1.0)
            res = minimize_scalar(rt_loss, bounds=bounds, method='bounded', options={'xatol': tolerance})

            opt_alpha = float(res.x)
            pc_eff_opt = pc_h / opt_alpha
            sw_est_opt = np.interp(pc_eff_opt, pc_pts, sw_pts, left=1.0, right=sw_pts[-1])
            pure_opt_rmse = float(np.sqrt(np.mean((sw_est_opt - target_sw) ** 2)))
            init_rmse = float(np.sqrt(np.mean((np.interp(pc_h, pc_pts, sw_pts, left=1.0, right=sw_pts[-1]) - target_sw) ** 2)))

            optimal_multipliers[rt] = opt_alpha
            rt_results[rt] = {
                'initial_multiplier': 1.0,
                'optimal_multiplier': opt_alpha,
                'initial_rmse': init_rmse,
                'optimal_rmse': pure_opt_rmse,
                'rmse_reduction_pct': (1.0 - pure_opt_rmse / init_rmse) * 100.0 if init_rmse > 0 else 0.0,
                'iterations': res.nfev,
                'weighting_mode': weighting_mode,
                'loss_type': loss_type
            }

        sw_dyn_init = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, {rt: 1.0 for rt in self.active_rts}, eqlnum_array=self.eqlnum_active
        )
        sw_dyn_opt = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, optimal_multipliers, eqlnum_array=self.eqlnum_active
        )

        init_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_init, self.rt_active, weights=self.pv_weights)
        opt_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_opt, self.rt_active, weights=self.pv_weights)

        return {
            'optimal_multipliers': optimal_multipliers,
            'per_rock_type': rt_results,
            'initial_metrics': init_metrics,
            'optimal_metrics': opt_metrics,
            'sw_dynamic_initial': sw_dyn_init,
            'sw_dynamic_optimal': sw_dyn_opt,
            'weighting_mode': weighting_mode,
            'loss_type': loss_type
        }

    def optimize_endpoints(
        self,
        alpha_bounds: Tuple[float, float] = (0.01, 20.0),
        pce_bounds: Tuple[float, float] = (0.0, 2.0),
        beta_bounds: Tuple[float, float] = (0.1, 5.0),
        optimize_entry_pressure: bool = True,
        optimize_curvature: bool = True,
        algorithm: str = "differential_evolution",
        asymptotic_guard: bool = True,
        swirr_tolerance: float = 0.20,
        min_alpha_guard: float = 0.80,
        weighting_mode: str = "Pore Volume (PV-Weighted)",
        loss_type: str = "Huber Loss (Robust to Scatter)",
        huber_delta: float = 0.10
    ) -> Dict[str, Any]:
        """
        Stage 2 & 3: Joint Multiplier, Endpoint, and Curvature Shape Optimization.
        Simultaneously optimizes:
        1. Vertical Pc multiplier: alpha_r
        2. Connate / irreducible water saturation endpoint: Sw_irr,r
        3. Entry / displacement pressure endpoint: Pce,r
        4. Curvature / pore-size distribution shape exponent: beta_r (Option A)
        
        Features:
        - Pore-Volume Weighting: Directly matches reservoir In-Place hydrocarbon volumes (HCPV).
        - Huber Loss: Immunizes the optimizer against extreme scatter and outliers (e.g. on RT-4).
        - algorithm: 'differential_evolution' (Global genetic metaheuristic with L-BFGS-B polish),
                     'dual_annealing' (Global simulated annealing), or
                     'L-BFGS-B' (Fast gradient-based local solver).
        - asymptotic_guard: Anchors Sw_irr within +-swirr_tolerance (default +-20%) of the core
                            lab measurement and enforces alpha >= min_alpha_guard to guarantee
                            that the steep physical asymptotic rise is preserved, completely
                            preventing unphysical Asymptotic Collapse (e.g. on RT-4).
        """
        optimal_params = {}
        rt_results = {}
        use_pv = "pore volume" in weighting_mode.lower() or "pv" in weighting_mode.lower()

        pc_hydro = self.equilibrator.compute_hydrostatic_pc(self.depth_active, eqlnum_array=self.eqlnum_active)

        for rt in self.active_rts:
            mask = (self.rt_active == rt)
            if not np.any(mask):
                continue

            pc_h = pc_hydro[mask]
            target_sw = self.sw_static_active[mask]
            cell_w = self.pv_weights[mask] if use_pv else np.ones(np.sum(mask), dtype=np.float32)
            table = self.equilibrator.scal_manager.tables[rt]

            base_sw = table.base_sw
            base_swirr = table.base_swirr
            base_pc = table.base_pcow
            s_wn = np.clip((base_sw - base_swirr) / (1.0 - base_swirr), 0.0, 1.0)

            # Determine petrophysical parameter bounds
            if asymptotic_guard:
                # Petrophysical anchoring: keep Swirr strictly within +- swirr_tolerance of core lab measurement
                swirr_min = max(0.01, float(base_swirr * (1.0 - swirr_tolerance)))
                swirr_max = min(0.98, float(base_swirr * (1.0 + swirr_tolerance)))
                # Guard alpha to ensure vertical asymptotic rise isn't destroyed
                eff_alpha_min = max(min_alpha_guard, alpha_bounds[0])
                eff_alpha_bounds = (eff_alpha_min, max(eff_alpha_min * 2.0, alpha_bounds[1]))
            else:
                target_min = float(np.min(target_sw))
                target_mean = float(np.mean(target_sw))
                swirr_min = max(0.001, target_min * 0.5)
                swirr_max = min(0.95, max(target_mean, base_swirr))
                eff_alpha_bounds = alpha_bounds

            if optimize_curvature:
                # 4 parameters: [alpha, swirr, pce, beta]
                x0 = [max(1.0, eff_alpha_bounds[0]), base_swirr, 0.0, 1.0]
                bnds = [eff_alpha_bounds, (swirr_min, swirr_max), pce_bounds, beta_bounds]

                def obj_joint(p):
                    a_val, sw_val, pce_val, beta_val = p
                    s_wn_curv = s_wn ** beta_val
                    new_sw = sw_val + s_wn_curv * (1.0 - sw_val)
                    sc_pc = pce_val * (1.0 - s_wn_curv) + base_pc * a_val
                    sc_pc[-1] = 0.0
                    sc_pc = np.maximum(0.0, sc_pc)
                    for i in range(len(sc_pc) - 2, -1, -1):
                        if sc_pc[i] < sc_pc[i + 1]:
                            sc_pc[i] = sc_pc[i + 1]
                    sw_est = np.interp(pc_h, sc_pc[::-1], new_sw[::-1], left=1.0, right=new_sw[0])
                    loss_val = compute_residual_loss(sw_est - target_sw, cell_w, loss_type=loss_type, delta=huber_delta)
                    if asymptotic_guard:
                        pen_sw = 0.25 * ((sw_val - base_swirr) / base_swirr) ** 2
                        pen_a = 0.25 * (max(0.0, 1.0 - a_val)) ** 2
                        return loss_val + pen_sw + pen_a
                    return loss_val

            elif optimize_entry_pressure:
                # 3 parameters: [alpha, swirr, pce]
                x0 = [max(1.0, eff_alpha_bounds[0]), base_swirr, 0.0]
                bnds = [eff_alpha_bounds, (swirr_min, swirr_max), pce_bounds]

                def obj_joint(p):
                    a_val, sw_val, pce_val = p
                    new_sw = sw_val + s_wn * (1.0 - sw_val)
                    sc_pc = pce_val * (1.0 - s_wn) + base_pc * a_val
                    sc_pc[-1] = 0.0
                    sc_pc = np.maximum(0.0, sc_pc)
                    for i in range(len(sc_pc) - 2, -1, -1):
                        if sc_pc[i] < sc_pc[i + 1]:
                            sc_pc[i] = sc_pc[i + 1]
                    sw_est = np.interp(pc_h, sc_pc[::-1], new_sw[::-1], left=1.0, right=new_sw[0])
                    loss_val = compute_residual_loss(sw_est - target_sw, cell_w, loss_type=loss_type, delta=huber_delta)
                    if asymptotic_guard:
                        pen_sw = 0.25 * ((sw_val - base_swirr) / base_swirr) ** 2
                        pen_a = 0.25 * (max(0.0, 1.0 - a_val)) ** 2
                        return loss_val + pen_sw + pen_a
                    return loss_val
            else:
                # 2 parameters: [alpha, swirr]
                x0 = [max(1.0, eff_alpha_bounds[0]), base_swirr]
                bnds = [eff_alpha_bounds, (swirr_min, swirr_max)]

                def obj_joint(p):
                    a_val, sw_val = p
                    new_sw = sw_val + s_wn * (1.0 - sw_val)
                    sc_pc = base_pc * a_val
                    sc_pc[-1] = 0.0
                    sc_pc = np.maximum(0.0, sc_pc)
                    for i in range(len(sc_pc) - 2, -1, -1):
                        if sc_pc[i] < sc_pc[i + 1]:
                            sc_pc[i] = sc_pc[i + 1]
                    sw_est = np.interp(pc_h, sc_pc[::-1], new_sw[::-1], left=1.0, right=new_sw[0])
                    loss_val = compute_residual_loss(sw_est - target_sw, cell_w, loss_type=loss_type, delta=huber_delta)
                    if asymptotic_guard:
                        pen_sw = 0.25 * ((sw_val - base_swirr) / base_swirr) ** 2
                        pen_a = 0.25 * (max(0.0, 1.0 - a_val)) ** 2
                        return loss_val + pen_sw + pen_a
                    return loss_val

            init_rmse = float(np.sqrt(np.mean((np.interp(pc_h, base_pc[::-1], base_sw[::-1], left=1.0, right=base_sw[0]) - target_sw) ** 2)))
            algo_key = algorithm.lower().replace("-", "").replace("_", "")

            if "differential" in algo_key or "evolution" in algo_key or algo_key == "de":
                res = differential_evolution(
                    obj_joint,
                    bounds=bnds,
                    seed=42,
                    maxiter=40,
                    popsize=12,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    polish=True
                )
            elif "dual" in algo_key or "annealing" in algo_key:
                res = dual_annealing(
                    obj_joint,
                    bounds=bnds,
                    seed=42,
                    maxiter=120
                )
            else:  # L-BFGS-B or local default
                res = minimize(obj_joint, x0, bounds=bnds, method='L-BFGS-B')

            if optimize_curvature:
                opt_a, opt_sw, opt_pce, opt_beta = res.x
            elif optimize_entry_pressure:
                opt_a, opt_sw, opt_pce = res.x
                opt_beta = 1.0
            else:
                opt_a, opt_sw = res.x
                opt_pce = 0.0
                opt_beta = 1.0

            # Pure unregularized RMSE evaluation for clean petrophysical reporting
            s_wn_curv = s_wn ** opt_beta
            new_sw_eval = opt_sw + s_wn_curv * (1.0 - opt_sw)
            sc_pc_eval = opt_pce * (1.0 - s_wn_curv) + base_pc * opt_a
            sc_pc_eval[-1] = 0.0
            sc_pc_eval = np.maximum(0.0, sc_pc_eval)
            for i in range(len(sc_pc_eval) - 2, -1, -1):
                if sc_pc_eval[i] < sc_pc_eval[i + 1]:
                    sc_pc_eval[i] = sc_pc_eval[i + 1]
            sw_est_opt = np.interp(pc_h, sc_pc_eval[::-1], new_sw_eval[::-1], left=1.0, right=new_sw_eval[0])
            pure_opt_rmse = float(np.sqrt(np.mean((sw_est_opt - target_sw) ** 2)))

            optimal_params[rt] = {
                'alpha': float(opt_a),
                'swirr': float(opt_sw),
                'pce': float(opt_pce),
                'beta': float(opt_beta)
            }

            rt_results[rt] = {
                'base_swirr': base_swirr,
                'optimal_swirr': float(opt_sw),
                'optimal_alpha': float(opt_a),
                'optimal_pce': float(opt_pce),
                'optimal_beta': float(opt_beta),
                'initial_rmse': init_rmse,
                'optimal_rmse': pure_opt_rmse,
                'rmse_reduction_pct': (1.0 - pure_opt_rmse / init_rmse) * 100.0 if init_rmse > 0 else 0.0,
                'iterations': getattr(res, 'nfev', 0),
                'algorithm': algorithm,
                'asymptotic_guard': asymptotic_guard,
                'weighting_mode': weighting_mode,
                'loss_type': loss_type
            }

        sw_dyn_init = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, {rt: 1.0 for rt in self.active_rts}, eqlnum_array=self.eqlnum_active
        )
        sw_dyn_opt = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, optimal_params, eqlnum_array=self.eqlnum_active
        )

        init_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_init, self.rt_active, weights=self.pv_weights)
        opt_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_opt, self.rt_active, weights=self.pv_weights)

        return {
            'optimal_parameters': optimal_params,
            'per_rock_type': rt_results,
            'initial_metrics': init_metrics,
            'optimal_metrics': opt_metrics,
            'sw_dynamic_initial': sw_dyn_init,
            'sw_dynamic_optimal': sw_dyn_opt,
            'algorithm': algorithm,
            'asymptotic_guard': asymptotic_guard,
            'weighting_mode': weighting_mode,
            'loss_type': loss_type
        }
