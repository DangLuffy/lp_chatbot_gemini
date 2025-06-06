# /app/solver/utils.py
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import copy # Để tạo bản sao sâu của problem_data

logger = logging.getLogger(__name__)

def standardize_problem_for_simplex(
    problem_data_input: Dict[str, Any],
    logs: List[str]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Chuẩn hóa bài toán cho các bộ giải Simplex:
    1. Đảm bảo hàm mục tiêu là "min". Nếu là "max", chuyển đổi max Z thành min -Z.
    2. Đảm bảo tất cả các ràng buộc là "<=".
       - Ràng buộc ">=" được chuyển thành "<=" bằng cách nhân cả hai vế với -1.
       - Ràng buộc "==" được chuyển thành hai ràng buộc: một "<=" và một ">=" (sau đó ">=" lại được chuyển thành "<=").

    Args:
        problem_data_input: Dữ liệu bài toán ở "Định dạng A":
            {
                "objective": "min" | "max",
                "coeffs": [-3, -2],
                "variables_names_for_title_only": ["x1", "x2"],
                "constraints": [
                    {"name": "RB1", "lhs": [2, 1], "op": "<=", "rhs": 2}, ...
                ]
            }
        logs: List để ghi lại các bước xử lý.

    Returns:
        Một tuple (standardized_problem_data, was_maximized):
        - standardized_problem_data: Dữ liệu bài toán đã được chuẩn hóa.
        - was_maximized: True nếu hàm mục tiêu ban đầu là "maximize".
    """
    if not problem_data_input or not isinstance(problem_data_input, dict):
        logs.append("ERROR (standardize): Input problem_data is empty or not a dictionary.")
        return None, False

    # Tạo bản sao sâu để không làm thay đổi dữ liệu gốc
    problem_data = copy.deepcopy(problem_data_input)
    logs.append("Standardizing problem for Simplex solvers...")

    was_maximized = False
    original_objective_type = problem_data.get("objective", "min").lower()

    # 1. Chuẩn hóa hàm mục tiêu về "min"
    if original_objective_type == "maximize":
        was_maximized = True
        problem_data["objective"] = "min"
        original_coeffs = problem_data.get("coeffs", [])
        problem_data["coeffs"] = [-c for c in original_coeffs]
        logs.append(f"Objective converted from MAX to MIN. Coefficients multiplied by -1: {problem_data['coeffs']}")
    elif original_objective_type != "min":
        logs.append(f"ERROR (standardize): Invalid objective type '{original_objective_type}'. Expected 'min' or 'max'.")
        return None, False
    else:
        logs.append("Objective is already MIN.")

    # 2. Chuẩn hóa các ràng buộc về "<="
    standardized_constraints: List[Dict[str, Any]] = []
    original_constraints = problem_data.get("constraints", [])

    if not isinstance(original_constraints, list):
        logs.append("ERROR (standardize): 'constraints' field is not a list or is missing.")
        return None, was_maximized

    for i, constr in enumerate(original_constraints):
        if not isinstance(constr, dict) or not all(k in constr for k in ["lhs", "op", "rhs"]):
            logs.append(f"ERROR (standardize): Constraint at index {i} is not a valid dictionary or missing keys (lhs, op, rhs). Constraint: {constr}")
            return None, was_maximized

        op = constr.get("op")
        lhs = list(constr.get("lhs", [])) # Đảm bảo là list
        rhs = constr.get("rhs", 0.0)
        name = constr.get("name", f"c{i+1}")

        if op in ["<=", "≤"]:
            standardized_constraints.append(constr.copy()) # Giữ nguyên
            logs.append(f"Constraint '{name}': '{op}' is already '<=' type. Kept as is.")
        elif op in [">=", "≥"]:
            # Chuyển A >= B thành -A <= -B
            new_lhs = [-val for val in lhs]
            new_rhs = -rhs
            standardized_constraints.append({
                "name": name + "_geq_to_leq",
                "lhs": new_lhs,
                "op": "<=",
                "rhs": new_rhs
            })
            logs.append(f"Constraint '{name}': Converted from '{op}' to '<='. LHS and RHS multiplied by -1.")
        elif op in ["==", "="]:
            # Chuyển A == B thành A <= B và A >= B (tức là -A <= -B)
            # Ràng buộc 1: A <= B
            standardized_constraints.append({
                "name": name + "_eq_leq",
                "lhs": list(lhs), # Tạo bản sao
                "op": "<=",
                "rhs": rhs
            })
            # Ràng buộc 2: A >= B  => -A <= -B
            standardized_constraints.append({
                "name": name + "_eq_geq_to_leq",
                "lhs": [-val for val in lhs],
                "op": "<=",
                "rhs": -rhs
            })
            logs.append(f"Constraint '{name}': Converted from '{op}' to two '<=' constraints.")
        else:
            logs.append(f"ERROR (standardize): Unknown constraint operator '{op}' for constraint '{name}'.")
            return None, was_maximized

    problem_data["constraints"] = standardized_constraints
    logs.append("Problem standardization complete.")
    return problem_data, was_maximized


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s (utils.py): %(message)s')
    test_logs: List[str] = []

    # Test 1: Maximize problem
    problem_max = {
        "objective": "max",
        "coeffs": [3, 2],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "C1", "lhs": [1, 1], "op": "<=", "rhs": 10},
            {"name": "C2", "lhs": [2, 1], "op": ">=", "rhs": 5}  # Sẽ được chuyển đổi
        ]
    }
    print("\n--- Testing Standardization (Maximize problem) ---")
    standardized_max, was_max_max = standardize_problem_for_simplex(problem_max, test_logs)
    if standardized_max:
        print(f"Standardized Data (Max): {standardized_max}")
        print(f"Was Maximized (Max): {was_max_max}")
        # Không còn gọi convert_problem_to_matrix_form nữa
    print("\nLogs (Max Test):")
    for log in test_logs: print(log)
    test_logs.clear()

    # Test 2: Minimize problem with equality
    problem_min_eq = {
        "objective": "min",
        "coeffs": [5, 8],
        "variables_names_for_title_only": ["y1", "y2"],
        "constraints": [
            {"name": "CE1", "lhs": [1, 0], "op": "==", "rhs": 3}, # Sẽ được chuyển đổi
            {"name": "CE2", "lhs": [0, 1], "op": "<=", "rhs": 7}
        ]
    }
    print("\n--- Testing Standardization (Minimize problem with equality) ---")
    standardized_min_eq, was_max_min_eq = standardize_problem_for_simplex(problem_min_eq, test_logs)
    if standardized_min_eq:
        print(f"Standardized Data (Min Eq): {standardized_min_eq}")
        print(f"Was Maximized (Min Eq): {was_max_min_eq}")
        # Không còn gọi convert_problem_to_matrix_form nữa
    print("\nLogs (Min Eq Test):")
    for log in test_logs: print(log)
    test_logs.clear()

    # Test 3: Problem with no constraints
    problem_no_constraints = {
        "objective": "min",
        "coeffs": [1, 1],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": []
    }
    print("\n--- Testing Standardization (No constraints) ---")
    standardized_no_constr, _ = standardize_problem_for_simplex(problem_no_constraints, test_logs)
    if standardized_no_constr:
        print(f"Standardized Data (No Constr): {standardized_no_constr}")
        # Không còn gọi convert_problem_to_matrix_form nữa
    print("\nLogs (No Constr Test):")
    for log in test_logs: print(log)
    test_logs.clear()
