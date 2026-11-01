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
from scipy.optimize import minimize_scalar, minimize
from .equilibrator import ReservoirEquilibrator


class OptimizationMetrics:
    """
    Evaluator for spatial saturation matching quality.
    """

    @staticmethod
    def compute(
        sw_static: np.ndarray,
        sw_dynamic: np.ndarray,
        rt_array: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive spatial mismatch metrics between static and dynamic Sw.
        
        Metrics:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - Max Error
        - Bias (Mean Signed Difference: Sw_dyn - Sw_stat)
        - R2 (Coefficient of Determination)
        - Per-Rock-Type RMSE and MAE
        """
        diff = sw_dynamic - sw_static
        sq_diff = diff ** 2
        abs_diff = np.abs(diff)

        n = len(sw_static)
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

        if rt_array is not None:
            unique_rts = np.unique(rt_array)
            for rt in unique_rts:
                if rt == 0:
                    continue
                mask = (rt_array == rt)
                if np.any(mask):
                    metrics[f'rt_{int(rt)}_rmse'] = float(np.sqrt(np.mean(sq_diff[mask])))
                    metrics[f'rt_{int(rt)}_mae'] = float(np.mean(abs_diff[mask]))

        return metrics


class PcOptimizer:
    """
    Automated optimizer for Rock-Type Capillary Pressure multipliers, endpoints, and curvature.
    """

    def __init__(
        self,
        equilibrator: ReservoirEquilibrator,
        depth_active: np.ndarray,
        rt_active: np.ndarray,
        sw_static_active: np.ndarray,
        eqlnum_active: Optional[np.ndarray] = None
    ):
        self.equilibrator = equilibrator
        self.depth_active = depth_active
        self.rt_active = rt_active
        self.sw_static_active = sw_static_active
        self.eqlnum_active = eqlnum_active
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
        tolerance: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Stage 1: Multiplier-only optimization using 1D bounded Brent search.
        Optimizes alpha_r for each rock type with fixed baseline endpoints.
        """
        optimal_multipliers = {}
        rt_results = {}

        pc_hydro = self.equilibrator.compute_hydrostatic_pc(self.depth_active, eqlnum_array=self.eqlnum_active)

        for rt in self.active_rts:
            mask = (self.rt_active == rt)
            if not np.any(mask):
                continue

            pc_h = pc_hydro[mask]
            target_sw = self.sw_static_active[mask]
            table = self.equilibrator.scal_manager.tables[rt]

            pc_pts = table.base_pcow[::-1]
            sw_pts = table.base_sw[::-1]

            def rt_loss(alpha: float) -> float:
                pc_eff = pc_h / alpha
                sw_est = np.interp(pc_eff, pc_pts, sw_pts, left=1.0, right=sw_pts[-1])
                return float(np.sqrt(np.mean((sw_est - target_sw) ** 2)))

            init_rmse = rt_loss(1.0)
            res = minimize_scalar(rt_loss, bounds=bounds, method='bounded', options={'xatol': tolerance})

            opt_alpha = float(res.x)
            opt_rmse = float(res.fun)

            optimal_multipliers[rt] = opt_alpha
            rt_results[rt] = {
                'initial_multiplier': 1.0,
                'optimal_multiplier': opt_alpha,
                'initial_rmse': init_rmse,
                'optimal_rmse': opt_rmse,
                'rmse_reduction_pct': (1.0 - opt_rmse / init_rmse) * 100.0 if init_rmse > 0 else 0.0,
                'iterations': res.nfev
            }

        sw_dyn_init = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, {rt: 1.0 for rt in self.active_rts}, eqlnum_array=self.eqlnum_active
        )
        sw_dyn_opt = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, optimal_multipliers, eqlnum_array=self.eqlnum_active
        )

        init_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_init, self.rt_active)
        opt_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_opt, self.rt_active)

        return {
            'optimal_multipliers': optimal_multipliers,
            'per_rock_type': rt_results,
            'initial_metrics': init_metrics,
            'optimal_metrics': opt_metrics,
            'sw_dynamic_initial': sw_dyn_init,
            'sw_dynamic_optimal': sw_dyn_opt
        }

    def optimize_endpoints(
        self,
        alpha_bounds: Tuple[float, float] = (0.01, 20.0),
        pce_bounds: Tuple[float, float] = (0.0, 2.0),
        beta_bounds: Tuple[float, float] = (0.1, 5.0),
        optimize_entry_pressure: bool = True,
        optimize_curvature: bool = True
    ) -> Dict[str, Any]:
        """
        Stage 2 & 3: Joint Multiplier, Endpoint, and Curvature Shape Optimization.
        Simultaneously optimizes:
        1. Vertical Pc multiplier: alpha_r
        2. Connate / irreducible water saturation endpoint: Sw_irr,r
        3. Entry / displacement pressure endpoint: Pce,r
        4. Curvature / pore-size distribution shape exponent: beta_r (Option A)
        
        This allows the dynamic model to adapt both to the height transition zone
        and the asymptotic irreducible saturation floor of the geological model.
        """
        optimal_params = {}
        rt_results = {}

        pc_hydro = self.equilibrator.compute_hydrostatic_pc(self.depth_active, eqlnum_array=self.eqlnum_active)

        for rt in self.active_rts:
            mask = (self.rt_active == rt)
            if not np.any(mask):
                continue

            pc_h = pc_hydro[mask]
            target_sw = self.sw_static_active[mask]
            table = self.equilibrator.scal_manager.tables[rt]

            base_sw = table.base_sw
            base_swirr = table.base_swirr
            base_pc = table.base_pcow
            s_wn = np.clip((base_sw - base_swirr) / (1.0 - base_swirr), 0.0, 1.0)

            target_min = float(np.min(target_sw))
            target_mean = float(np.mean(target_sw))
            swirr_min = max(0.001, target_min * 0.5)
            swirr_max = min(0.95, max(target_mean, base_swirr))

            if optimize_curvature:
                # 4 parameters: [alpha, swirr, pce, beta]
                x0 = [1.0, base_swirr, 0.0, 1.0]
                bnds = [alpha_bounds, (swirr_min, swirr_max), pce_bounds, beta_bounds]

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
                    return float(np.sqrt(np.mean((sw_est - target_sw) ** 2)))

            elif optimize_entry_pressure:
                # 3 parameters: [alpha, swirr, pce]
                x0 = [1.0, base_swirr, 0.0]
                bnds = [alpha_bounds, (swirr_min, swirr_max), pce_bounds]

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
                    return float(np.sqrt(np.mean((sw_est - target_sw) ** 2)))
            else:
                # 2 parameters: [alpha, swirr]
                x0 = [1.0, base_swirr]
                bnds = [alpha_bounds, (swirr_min, swirr_max)]

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
                    return float(np.sqrt(np.mean((sw_est - target_sw) ** 2)))

            init_rmse = obj_joint(x0)
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
                'optimal_rmse': float(res.fun),
                'rmse_reduction_pct': (1.0 - float(res.fun) / init_rmse) * 100.0 if init_rmse > 0 else 0.0,
                'iterations': res.nfev
            }

        sw_dyn_init = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, {rt: 1.0 for rt in self.active_rts}, eqlnum_array=self.eqlnum_active
        )
        sw_dyn_opt = self.equilibrator.calculate_dynamic_sw(
            self.depth_active, self.rt_active, optimal_params, eqlnum_array=self.eqlnum_active
        )

        init_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_init, self.rt_active)
        opt_metrics = OptimizationMetrics.compute(self.sw_static_active, sw_dyn_opt, self.rt_active)

        return {
            'optimal_parameters': optimal_params,
            'per_rock_type': rt_results,
            'initial_metrics': init_metrics,
            'optimal_metrics': opt_metrics,
            'sw_dynamic_initial': sw_dyn_init,
            'sw_dynamic_optimal': sw_dyn_opt
        }
