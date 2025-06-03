# /app/solver/dispatcher.py
import logging
from typing import Dict, Any, Tuple, Optional, Callable, List

from .pulp_cbc_solver import solve_with_pulp_cbc
from .simple_dictionary_solver import solve_with_simplex_manual
from .geometric_solver import solve_with_geometric_method # <<<--- IMPORT BỘ GIẢI MỚI