# /app/solver/dispatcher.py
import logging
from typing import Dict, Any, Tuple, Optional, Callable, List
# import functools # Không cần thiết nếu dùng lambda hoặc truyền trực tiếp

# Import các hàm bao bọc solver đã được cập nhật/đổi tên
from .pulp_cbc_solver import solve_with_pulp_cbc
from .geometric_solver import solve_with_geometric_method
from .simple_dictionary_solver import solve_with_simple_dictionary # Đã thay thế simplex_manual
from .simplex_bland_solver import solve_with_simplex_bland
# Giả sử file chứa AuxiliaryProblemSolver đã được đổi tên thành auxiliary_problem_solver.py
from .auxiliary_problem_solver import solve_with_auxiliary_problem_simplex

logger = logging.getLogger(__name__)

# Định nghĩa một kiểu cho hàm solver (nhận problem_data và tùy chọn max_iterations)
SolverFunction = Callable[[Dict[str, Any], int], Tuple[Optional[Dict[str, Any]], List[str]]]

# Ánh xạ tên solver sang hàm thực thi
# Các hàm này giờ đây nhận "Định dạng A" làm problem_data.
# Các hàm solver Simplex sẽ tự gọi standardize_problem_for_simplex bên trong.
AVAILABLE_SOLVERS: Dict[str, SolverFunction] = {
    "pulp_cbc": lambda pd, mi=0: solve_with_pulp_cbc(pd), # pulp_cbc không dùng max_iterations từ dispatcher
    "geometric": lambda pd, mi=0: solve_with_geometric_method(pd), # geometric không dùng max_iterations từ dispatcher
    "simple_dictionary": solve_with_simple_dictionary, # Hàm này có max_iterations=50 làm mặc định
    "simplex_bland": solve_with_simplex_bland,       # Hàm này có max_iterations=50 làm mặc định
    "auxiliary": solve_with_auxiliary_problem_simplex, # Hàm này có max_iterations_total=50 làm mặc định
    # "two_phase": solve_with_two_phase_simplex, # Nếu bạn có solver Two-Phase chuẩn riêng
    # "dual_simplex": solve_with_dual_simplex,   # Nếu bạn có solver Đối ngẫu
}

def dispatch_solver(
    problem_data: Dict[str, Any], # Sẽ nhận "Định dạng A"
    solver_name: str = "pulp_cbc",
    max_iterations: int = 50 # Tham số này sẽ được truyền cho các solver Simplex
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Gọi solver được chỉ định để giải bài toán LP.

    Args:
        problem_data: Dictionary chứa thông tin bài toán (mong đợi "Định dạng A").
        solver_name: Tên của solver cần sử dụng.
        max_iterations: Số vòng lặp tối đa cho các solver lặp.

    Returns:
        Một tuple (solution, logs) từ solver được chọn.
    """
    logs = [f"Dispatcher: Attempting to use solver '{solver_name}' with max_iterations={max_iterations}."]
    logger.info(f"Dispatching to solver: {solver_name}, problem_data keys: {list(problem_data.keys())}")

    solver_func_base = AVAILABLE_SOLVERS.get(solver_name)

    if solver_func_base:
        try:
            # Truyền max_iterations cho các solver Simplex.
            # pulp_cbc và geometric được gọi qua lambda nên không nhận max_iterations từ đây.
            if solver_name in ["simple_dictionary", "simplex_bland", "auxiliary"]:
                solution, solver_logs = solver_func_base(problem_data, max_iterations)
            else: # Cho pulp_cbc, geometric (lambda đã xử lý việc không cần max_iterations)
                solution, solver_logs = solver_func_base(problem_data, 0) # Số 0 chỉ là placeholder

            logs.extend(solver_logs)
            if solution is None and not any("Error" in log.lower() or "failed" in log.lower() for log in solver_logs):
                 logs.append(f"Dispatcher: Solver '{solver_name}' did not return a solution object, but no explicit error was logged by it.")
            elif solution:
                 logs.append(f"Dispatcher: Solver '{solver_name}' completed.")
            return solution, logs
        except TypeError as te:
            error_msg = f"Dispatcher: TypeError calling solver '{solver_name}'. It might not accept the provided arguments (e.g. max_iterations). Error: {te}"
            logs.append(f"Error: {error_msg}")
            logger.exception(error_msg)
            return None, logs
        except Exception as e:
            error_msg = f"Dispatcher: An unexpected error occurred while running solver '{solver_name}': {e}"
            logs.append(f"Error: {error_msg}")
            logger.exception(error_msg)
            return None, logs
    else:
        error_msg = f"Solver '{solver_name}' not found. Available solvers: {list(AVAILABLE_SOLVERS.keys())}"
        logs.append(f"Error: {error_msg}")
        logger.error(error_msg)
        return None, logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ví dụ sử dụng "Định dạng A"
    sample_problem_A = {
        "objective": "maximize",
        "coeffs": [3, 2],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "lhs": [1, 1], "op": "<=", "rhs": 4},
            {"name": "c2", "lhs": [2, 1], "op": "<=", "rhs": 5},
            {"name": "nonneg_x1", "lhs": [1,0], "op": ">=", "rhs": 0},
            {"name": "nonneg_x2", "lhs": [0,1], "op": ">=", "rhs": 0},
        ]
    }

    print("--- Dispatching to PuLP CBC (Default, Format A) ---")
    solution_pulp, logs_pulp = dispatch_solver(sample_problem_A)
    if solution_pulp: print("Solution (PuLP CBC):", solution_pulp)

    print("\n--- Dispatching to Simple Dictionary (Format A) ---")
    solution_simple, logs_simple = dispatch_solver(sample_problem_A, solver_name="simple_dictionary", max_iterations=15)
    if solution_simple: print("Solution (Simple Dictionary):", solution_simple)

    print("\n--- Dispatching to Simplex Bland (Format A) ---")
    solution_bland, logs_bland = dispatch_solver(sample_problem_A, solver_name="simplex_bland", max_iterations=15)
    if solution_bland: print("Solution (Simplex Bland):", solution_bland)

    problem_f72cfa_A = {
        "objective": "min",
        "coeffs": [1, 1],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "R1_orig", "lhs": [-1, 1], "op": "<=", "rhs": -2},
            {"name": "R2_orig", "lhs": [1, 2], "op": "<=", "rhs": 4},
            {"name": "R3_orig", "lhs": [1, 0], "op": "<=", "rhs": 1},
            {"name": "nonneg_x1", "lhs": [1,0], "op": ">=", "rhs": 0},
            {"name": "nonneg_x2", "lhs": [0,1], "op": ">=", "rhs": 0}
        ]
    }
    print("\n--- Dispatching to Auxiliary Solver (Format A) ---")
    solution_aux, logs_aux = dispatch_solver(problem_f72cfa_A, solver_name="auxiliary", max_iterations=25)
    if solution_aux: print("Solution (Auxiliary):", solution_aux)

    print("\n--- Dispatching to Geometric (Format A) ---")
    problem_geom_A = {
        "objective": "maximize", "coeffs": [7,3],
        "variables_names_for_title_only": ["chair", "table"],
        "constraints": [
            {"name":"Wood", "lhs": [2,3], "op":"<=", "rhs":18},
            {"name":"Labor", "lhs": [1,1], "op":"<=", "rhs":8},
            {"name":"nonneg_chair", "lhs": [1,0], "op":">=", "rhs":0},
            {"name":"nonneg_table", "lhs": [0,1], "op":">=", "rhs":0},
        ]
    }
    solution_geom, logs_geom = dispatch_solver(problem_geom_A, solver_name="geometric")
    if solution_geom: print(f"Solution (Geometric): Status={solution_geom.get('status')}, Obj={solution_geom.get('objective_value')}, Vars={solution_geom.get('variables')}")

    print("\n--- Dispatching to NonExistent Solver ---")
    solution_none, logs_none = dispatch_solver(sample_problem_A, solver_name="nonexistent_solver")
    if not solution_none: print("Solver not found as expected.")

