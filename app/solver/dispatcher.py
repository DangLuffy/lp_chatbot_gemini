# /app/solver/dispatcher.py
import logging
from typing import Dict, Any, Tuple, Optional, Callable, List

from .pulp_cbc_solver import solve_with_pulp_cbc
from .simple_dictionary_solver import solve_with_simplex_manual
from .geometric_solver import solve_with_geometric_method # <<<--- IMPORT BỘ GIẢI MỚI
# Khi có thêm solver, import chúng ở đây
# from .another_solver import solve_with_another_method

logger = logging.getLogger(__name__)

# Định nghĩa một kiểu cho hàm solver
SolverFunction = Callable[[Dict[str, Any]], Tuple[Optional[Dict[str, Any]], List[str]]]
# Nếu solver có thêm tham số (ví dụ: max_iterations), cần điều chỉnh Callable
# Hoặc sử dụng functools.partial để bao bọc solver với các tham số mặc định.

# Ánh xạ tên solver sang hàm thực thi
# Ánh xạ tên solver sang hàm thực thi
AVAILABLE_SOLVERS: Dict[str, SolverFunction] = {
    "pulp_cbc": solve_with_pulp_cbc,
    "simplex_manual": lambda pd: solve_with_simplex_manual(pd, max_iterations=50),
    "geometric": solve_with_geometric_method, # <<<--- THÊM BỘ GIẢI MỚI VÀO DICTIONARY
}

def dispatch_solver(
    problem_data: Dict[str, Any],
    solver_name: str = "pulp_cbc" # Mặc định dùng pulp_cbc
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Gọi solver được chỉ định để giải bài toán LP.

    Args:
        problem_data: Dictionary chứa thông tin bài toán.
        solver_name: Tên của solver cần sử dụng (ví dụ: "pulp_cbc", "simplex_manual").

    Returns:
        Một tuple (solution, logs) từ solver được chọn.
        Trả về (None, ["Error: Solver not found."]) nếu tên solver không hợp lệ.
    """
    logs = [f"Dispatcher: Attempting to use solver '{solver_name}'."]
    logger.info(f"Dispatching to solver: {solver_name}")

    solver_func = AVAILABLE_SOLVERS.get(solver_name)

    if solver_func:
        try:
            solution, solver_logs = solver_func(problem_data)
            logs.extend(solver_logs)
            if solution is None and not any("Error" in log for log in solver_logs):
                 logs.append(f"Dispatcher: Solver '{solver_name}' did not return a solution object, but no explicit error was logged by it.")
            elif solution:
                 logs.append(f"Dispatcher: Solver '{solver_name}' completed.")
            return solution, logs
        except Exception as e:
            error_msg = f"Dispatcher: An unexpected error occurred while running solver '{solver_name}': {e}"
            logs.append(f"Error: {error_msg}")
            logger.exception(error_msg) # Ghi lại traceback
            return None, logs
    else:
        error_msg = f"Solver '{solver_name}' not found. Available solvers: {list(AVAILABLE_SOLVERS.keys())}"
        logs.append(f"Error: {error_msg}")
        logger.error(error_msg)
        return None, logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sample_problem = {
        "objective": {"type": "maximize", "coefficients": [3, 2]},
        "variables": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 1], "type": "<=", "rhs": 4},
            {"name": "c2", "coefficients": [2, 1], "type": "<=", "rhs": 5},
        ],
        "bounds": {"x1": {"low": 0}, "x2": {"low": 0}}
    }

    print("--- Dispatching to PuLP CBC (Default) ---")
    solution_pulp, logs_pulp = dispatch_solver(sample_problem)
    for log in logs_pulp:
        print(log)
    if solution_pulp:
        print("Solution (PuLP CBC):", solution_pulp)

    print("\n--- Dispatching to Simplex Manual ---")
    # Lưu ý: simplex_manual_solver hiện tại rất cơ bản
    solution_manual, logs_manual = dispatch_solver(sample_problem, solver_name="simplex_manual")
    for log in logs_manual:
        print(log)
    if solution_manual:
        print("Solution (Simplex Manual):", solution_manual)

    print("\n--- Dispatching to NonExistent Solver ---")
    solution_none, logs_none = dispatch_solver(sample_problem, solver_name="nonexistent_solver")
    for log in logs_none:
        print(log)
    if solution_none: # Sẽ không có solution
        print("Solution (NonExistent):", solution_none)
