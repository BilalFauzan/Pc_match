"""
pc_matching
===========
Automated Static-Dynamic Initial Water Saturation (Sw) Distribution Matching
and Capillary Pressure (Pc-Sw) Optimization.
"""

from .grdecl_io import FastGRDECLParser, extract_active_reservoir
from .scal_manager import SCALTable, SCALManager
from .equilibrator import EquilibrationParams, ReservoirEquilibrator
from .optimizer import OptimizationMetrics, PcOptimizer
from .simulation_io import TNavWorkflowManager
from .visualizer import (
    plot_scal_comparison,
    plot_saturation_height,
    plot_crossplots,
    plot_error_distributions
)

__all__ = [
    "FastGRDECLParser",
    "extract_active_reservoir",
    "SCALTable",
    "SCALManager",
    "EquilibrationParams",
    "ReservoirEquilibrator",
    "OptimizationMetrics",
    "PcOptimizer",
    "TNavWorkflowManager",
    "plot_scal_comparison",
    "plot_saturation_height",
    "plot_crossplots",
    "plot_error_distributions"
]

