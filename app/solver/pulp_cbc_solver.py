# /app/solver/pulp_cbc_solver.py

import pulp
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def solve_with_pulp_cbc(problem_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Giải bài toán Quy hoạch Tuyến tính (LP) sử dụng PuLP với bộ giải CBC.
    Hàm này giờ đây mong đợi problem_data ở "Định dạng A".

    Args:
        problem_data: Một dictionary chứa thông tin bài toán ở "Định dạng A":
                      {
                          "objective": "maximize" or "minimize",
                          "coeffs": [c1, c2, ...], # Hệ số hàm mục tiêu
                          "variables_names_for_title_only": ["x1", "x2", ...], # Tên các biến
                          "constraints": [
                              {
                                  "name": "constraint_1",
                                  "lhs": [a11, a12, ...], # Hệ số vế trái
                                  "op": "<=", ">=", or "==", # Toán tử
                                  "rhs": b1
                              },
                              ...
                          ],
                          "bounds": { # Tùy chọn, nếu không có sẽ giả định biến không âm
                              "x1": {"low": 0, "up": None},
                              ...
                          }
                      }

    Returns:
        Một tuple (solution, logs):
        - solution: Dictionary chứa kết quả.
        - logs: Danh sách các chuỗi log.
    """
    logs = ["Starting PuLP CBC solver with Format A input."]
    try:
        # 1. Kiểm tra dữ liệu đầu vào theo "Định dạng A"
        required_keys = ["objective", "coeffs", "variables_names_for_title_only", "constraints"]
        if not all(k in problem_data for k in required_keys):
            missing_keys = [k for k in required_keys if k not in problem_data]
            logs.append(f"Error: Missing required fields in problem_data (Format A): {missing_keys}.")
            logger.error(f"Missing required fields for Format A: {missing_keys}")
            return None, logs

        variable_names = problem_data["variables_names_for_title_only"]
        obj_coeffs = problem_data["coeffs"]
        num_vars = len(variable_names)

        if len(obj_coeffs) != num_vars:
            logs.append(f"Error: Number of objective coefficients ({len(obj_coeffs)}) does not match number of variables ({num_vars}).")
            logger.error("Objective coefficients mismatch with variable names.")
            return None, logs

        for i, constr in enumerate(problem_data.get("constraints", [])):
            if not isinstance(constr, dict) or not all(k in constr for k in ["lhs", "op", "rhs"]):
                logs.append(f"Error: Constraint at index {i} is malformed (missing lhs, op, or rhs).")
                logger.error(f"Malformed constraint at index {i}: {constr}")
                return None, logs
            if len(constr["lhs"]) != num_vars:
                logs.append(f"Error: Number of coefficients in constraint '{constr.get('name', f'Unnamed_{i}')}' ({len(constr['lhs'])}) does not match number of variables ({num_vars}).")
                logger.error(f"Constraint coefficients mismatch for {constr.get('name', f'Unnamed_{i}')}.")
                return None, logs

        # 2. Tạo bài toán LP
        objective_type_str = problem_data["objective"].lower()
        if objective_type_str == "maximize":
            prob = pulp.LpProblem("LP_Problem_PuLP", pulp.LpMaximize)
            logs.append("Problem type: Maximize.")
        elif objective_type_str == "minimize":
            prob = pulp.LpProblem("LP_Problem_PuLP", pulp.LpMinimize)
            logs.append("Problem type: Minimize.")
        else:
            logs.append(f"Error: Invalid objective type '{objective_type_str}'. Must be 'maximize' or 'minimize'.")
            logger.error(f"Invalid objective type: {objective_type_str}")
            return None, logs

        # 3. Định nghĩa các biến quyết định
        lp_variables = {}
        bounds_data = problem_data.get("bounds", {}) # Phần bounds vẫn giữ nguyên cấu trúc

        for var_name in variable_names:
            var_bounds = bounds_data.get(var_name, {})
            low_bound = var_bounds.get("low")
            up_bound = var_bounds.get("up")

            # Mặc định biến không âm nếu không có thông tin bound cụ thể cho biến đó
            if low_bound is None and var_name not in bounds_data:
                 low_bound = 0 # Đảm bảo không âm nếu không có bound được chỉ định

            lp_variables[var_name] = pulp.LpVariable(var_name, lowBound=low_bound, upBound=up_bound)
            logs.append(f"Defined variable: {var_name} (Low: {low_bound}, Up: {up_bound})")


        # 4. Định nghĩa hàm mục tiêu
        # obj_coeffs đã được lấy từ problem_data["coeffs"]
        prob += pulp.lpSum([obj_coeffs[i] * lp_variables[variable_names[i]] for i in range(num_vars)]), "Objective_Function"
        logs.append(f"Defined objective function: {pulp.lpSum([obj_coeffs[i] * lp_variables[variable_names[i]] for i in range(num_vars)])}")

        # 5. Định nghĩa các ràng buộc
        for i, constr_data in enumerate(problem_data["constraints"]):
            constr_lhs_coeffs = constr_data["lhs"] # Sử dụng "lhs"
            constr_op = constr_data["op"]       # Sử dụng "op"
            constr_rhs_val = constr_data["rhs"]
            constr_name_str = constr_data.get("name", f"Constraint_{i+1}")

            expr = pulp.lpSum([constr_lhs_coeffs[j] * lp_variables[variable_names[j]] for j in range(num_vars)])

            if constr_op == "<=" or constr_op == "≤":
                prob += expr <= constr_rhs_val, constr_name_str
            elif constr_op == ">=" or constr_op == "≥":
                prob += expr >= constr_rhs_val, constr_name_str
            elif constr_op == "==" or constr_op == "=":
                prob += expr == constr_rhs_val, constr_name_str
            else:
                logs.append(f"Error: Invalid constraint operator '{constr_op}' for constraint '{constr_name_str}'.")
                logger.error(f"Invalid constraint operator: {constr_op}")
                return None, logs
            logs.append(f"Defined constraint: {constr_name_str}: {expr} {constr_op} {constr_rhs_val}")

        logs.append("LP problem defined successfully.")
        # logs.append(f"Problem formulation:\n{prob}") # Có thể rất dài, bỏ comment nếu cần debug

        # 6. Giải bài toán
        logs.append("Attempting to solve with CBC solver...")
        status = prob.solve() # Mặc định msg=False
        logs.append(f"Solver status: {pulp.LpStatus[status]}")

        # 7. Xử lý kết quả
        if status == pulp.LpStatusOptimal:
            solution = {
                "status": "Optimal",
                "objective_value": pulp.value(prob.objective),
                "variables": {}
            }
            for v_obj in prob.variables(): # Đổi tên biến lặp
                solution["variables"][v_obj.name] = v_obj.varValue
            logs.append(f"Optimal solution found. Objective value: {solution['objective_value']}")
            logs.append(f"Variable values: {solution['variables']}")
            return solution, logs
        elif status == pulp.LpStatusInfeasible:
            logs.append("Problem is Infeasible. No solution exists.")
            return {"status": "Infeasible"}, logs
        elif status == pulp.LpStatusUnbounded:
            logs.append("Problem is Unbounded.")
            return {"status": "Unbounded"}, logs
        else: # Not Solved, Undefined
            logs.append(f"Solver did not find an optimal solution. Status: {pulp.LpStatus[status]}")
            return {"status": pulp.LpStatus[status]}, logs

    except Exception as e:
        error_msg = f"An error occurred during PuLP CBC solving: {e}"
        logs.append(f"Error: {error_msg}")
        logger.exception(error_msg)
        return None, logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Ví dụ sử dụng với "Định dạng A"
    sample_problem_A_format_max = {
        "objective": "maximize",
        "coeffs": [3, 2],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "lhs": [1, 1], "op": "<=", "rhs": 4},
            {"name": "c2", "lhs": [2, 1], "op": "<=", "rhs": 5},
        ],
         "bounds": {"x1": {"low": 0}, "x2": {"low": 0}} # x1 >=0, x2 >= 0
    }
    # Expected: x1=1, x2=3, Z=9 OR x1=2.5, x2=0, Z=7.5.
    # Với ràng buộc 2x1+x2 <=5 và x1+x2<=4, điểm cắt (1,3) Z=9. Đỉnh (2.5,0) Z=7.5. Đỉnh (0,4) Z=8.
    # Vậy Z=9 tại x1=1, x2=3 là tối ưu.

    sample_problem_A_format_min_geq = {
        "objective": "minimize",
        "coeffs": [7, 5],
        "variables_names_for_title_only": ["apple", "banana"],
        "constraints": [
            {"name": "ProteinReq", "lhs": [3, 2], "op": ">=", "rhs": 12}, # 3*apple + 2*banana >= 12
            {"name": "CarbReq", "lhs": [1, 4], "op": ">=", "rhs": 8}    # 1*apple + 4*banana >= 8
        ],
        "bounds": {"apple": {"low": 0}, "banana": {"low": 0}}
    }
    # Lời giải: apple=3.2, banana=1.2, Z_min = 7*3.2 + 5*1.2 = 22.4 + 6 = 28.4 (Theo online solver)

    print("--- Solving Maximization Problem (Format A) ---")
    solution_max, logs_max = solve_with_pulp_cbc(sample_problem_A_format_max)
    for log in logs_max: print(log)
    if solution_max: print("Solution:", solution_max)

    print("\n--- Solving Minimization Problem with GEQ (Format A) ---")
    solution_min, logs_min = solve_with_pulp_cbc(sample_problem_A_format_min_geq)
    for log in logs_min: print(log)
    if solution_min: print("Solution:", solution_min)

    # Test trường hợp thiếu key
    problem_missing_keys = {
        "objective": "maximize",
        "coeffs": [1,1]
        # Thiếu variables_names_for_title_only và constraints
    }
    print("\n--- Solving Problem with Missing Keys (Format A) ---")
    solution_missing, logs_missing = solve_with_pulp_cbc(problem_missing_keys)
    for log in logs_missing: print(log)
    if solution_missing: print("Solution:", solution_missing)


    # Test trường hợp số lượng hệ số không khớp
    problem_coeff_mismatch = {
        "objective": "maximize",
        "coeffs": [1,1,1], # 3 hệ số
        "variables_names_for_title_only": ["x1", "x2"], # 2 biến
        "constraints": []
    }
    print("\n--- Solving Problem with Coefficient Mismatch (Format A) ---")
    solution_mismatch, logs_mismatch = solve_with_pulp_cbc(problem_coeff_mismatch)
    for log in logs_mismatch: print(log)
    if solution_mismatch: print("Solution:", solution_mismatch)

