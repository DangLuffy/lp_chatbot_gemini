# /app/solver/pulp_cbc_solver.py

import pulp
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def solve_with_pulp_cbc(problem_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Giải bài toán Quy hoạch Tuyến tính (LP) sử dụng PuLP với bộ giải CBC.

    Args:
        problem_data: Một dictionary chứa thông tin bài toán.
                      Cấu trúc dự kiến:
                      {
                          "objective": {"type": "maximize" or "minimize", "coefficients": [c1, c2, ...]},
                          "variables": ["x1", "x2", ...], # Tên các biến
                          "constraints": [
                              {
                                  "name": "constraint_1",
                                  "coefficients": [a11, a12, ...],
                                  "type": "<=", ">=", or "==",
                                  "rhs": b1
                              },
                              ...
                          ],
                          "bounds": { # Tùy chọn, nếu không có sẽ giả định biến không âm
                              "x1": {"low": 0, "up": None},
                              "x2": {"low": 0, "up": 10},
                              ...
                          }
                      }

    Returns:
        Một tuple (solution, logs):
        - solution: Dictionary chứa kết quả (giá trị biến, giá trị hàm mục tiêu) nếu tìm thấy.
                    None nếu không có lời giải hoặc có lỗi.
        - logs: Danh sách các chuỗi log mô tả quá trình giải.
    """
    logs = ["Starting PuLP CBC solver."]
    try:
        # 1. Kiểm tra dữ liệu đầu vào (có thể thêm các bước kiểm tra chi tiết hơn)
        if not all(k in problem_data for k in ["objective", "variables", "constraints"]):
            logs.append("Error: Missing required fields in problem_data (objective, variables, constraints).")
            logger.error("Missing required fields in problem_data.")
            return None, logs

        num_vars = len(problem_data["variables"])
        if len(problem_data["objective"]["coefficients"]) != num_vars:
            logs.append("Error: Number of objective coefficients does not match number of variables.")
            logger.error("Objective coefficients mismatch.")
            return None, logs

        for constr in problem_data["constraints"]:
            if len(constr["coefficients"]) != num_vars:
                logs.append(f"Error: Number of coefficients in constraint '{constr.get('name', 'Unnamed')}' does not match number of variables.")
                logger.error(f"Constraint coefficients mismatch for {constr.get('name', 'Unnamed')}.")
                return None, logs

        # 2. Tạo bài toán LP
        objective_type = problem_data["objective"]["type"].lower()
        if objective_type == "maximize":
            prob = pulp.LpProblem("LP_Problem_PuLP", pulp.LpMaximize)
            logs.append("Problem type: Maximize.")
        elif objective_type == "minimize":
            prob = pulp.LpProblem("LP_Problem_PuLP", pulp.LpMinimize)
            logs.append("Problem type: Minimize.")
        else:
            logs.append(f"Error: Invalid objective type '{objective_type}'. Must be 'maximize' or 'minimize'.")
            logger.error(f"Invalid objective type: {objective_type}")
            return None, logs

        # 3. Định nghĩa các biến quyết định
        # Ví dụ: x1, x2, ...
        # Mặc định các biến là không âm (lowBound=0) trừ khi được chỉ định trong 'bounds'
        lp_variables = {}
        variable_names = problem_data["variables"]
        bounds_data = problem_data.get("bounds", {})

        for var_name in variable_names:
            var_bounds = bounds_data.get(var_name, {})
            low_bound = var_bounds.get("low") # Mặc định PuLP là -infinity nếu không có lowBound
            up_bound = var_bounds.get("up")   # Mặc định PuLP là +infinity nếu không có upBound

            # PuLP mặc định biến là continuous. Nếu cần biến nguyên (Integer) hoặc nhị phân (Binary),
            # cần thêm thông tin 'category' cho biến trong problem_data.
            # Ví dụ: "variables_details": {"x1": {"category": "Integer"}, "x2": {"category": "Binary"}}
            # var_category = problem_data.get("variables_details", {}).get(var_name, {}).get("category", "Continuous")
            # lp_variables[var_name] = pulp.LpVariable(var_name, lowBound=low_bound, upBound=up_bound, cat=var_category)

            # Mặc định là biến liên tục và không âm nếu không có bound.
            # Nếu low_bound không được cung cấp, PuLP coi là không có giới hạn dưới (có thể âm).
            # Để đảm bảo biến không âm theo mặc định của nhiều bài toán LP, ta đặt lowBound=0 nếu không có.
            if low_bound is None and var_name not in bounds_data: # Nếu không có trong bounds, mặc định không âm
                 low_bound = 0

            lp_variables[var_name] = pulp.LpVariable(var_name, lowBound=low_bound, upBound=up_bound)
            logs.append(f"Defined variable: {var_name} (Low: {low_bound}, Up: {up_bound})")


        # 4. Định nghĩa hàm mục tiêu
        obj_coeffs = problem_data["objective"]["coefficients"]
        prob += pulp.lpSum([obj_coeffs[i] * lp_variables[variable_names[i]] for i in range(num_vars)]), "Objective_Function"
        logs.append(f"Defined objective function: {pulp.lpSum([obj_coeffs[i] * lp_variables[variable_names[i]] for i in range(num_vars)])}")

        # 5. Định nghĩa các ràng buộc
        for i, constr_data in enumerate(problem_data["constraints"]):
            constr_coeffs = constr_data["coefficients"]
            constr_type = constr_data["type"]
            constr_rhs = constr_data["rhs"]
            constr_name = constr_data.get("name", f"Constraint_{i+1}")

            expr = pulp.lpSum([constr_coeffs[j] * lp_variables[variable_names[j]] for j in range(num_vars)])

            if constr_type == "<=":
                prob += expr <= constr_rhs, constr_name
            elif constr_type == ">=":
                prob += expr >= constr_rhs, constr_name
            elif constr_type == "==":
                prob += expr == constr_rhs, constr_name
            else:
                logs.append(f"Error: Invalid constraint type '{constr_type}' for constraint '{constr_name}'.")
                logger.error(f"Invalid constraint type: {constr_type}")
                return None, logs
            logs.append(f"Defined constraint: {constr_name}: {expr} {constr_type} {constr_rhs}")

        logs.append("LP problem defined successfully.")
        logs.append(f"Problem formulation:\n{prob}") # Ghi lại toàn bộ bài toán

        # 6. Giải bài toán
        logs.append("Attempting to solve with CBC solver...")
        # solver = pulp.PULP_CBC_CMD(msg=True) # msg=True để xem output của CBC
        # Bạn có thể chỉ định path tới CBC executable nếu cần:
        # solver = pulp.PULP_CBC_CMD(path='/path/to/cbc', msg=True)
        # Nếu CBC đã có trong PATH, không cần chỉ định path.
        # Để ẩn output của solver, dùng msg=False (mặc định)
        status = prob.solve()
        logs.append(f"Solver status: {pulp.LpStatus[status]}")

        # 7. Xử lý kết quả
        if status == pulp.LpStatusOptimal:
            solution = {
                "status": "Optimal",
                "objective_value": pulp.value(prob.objective),
                "variables": {}
            }
            for v in prob.variables():
                solution["variables"][v.name] = v.varValue
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
        logger.exception(error_msg) # Ghi lại cả traceback
        return None, logs

if __name__ == '__main__':
    # Ví dụ sử dụng
    logging.basicConfig(level=logging.INFO) # Để thấy output của logger

    sample_problem_maximize = {
        "objective": {"type": "maximize", "coefficients": [3, 2]}, # Maximize 3x1 + 2x2
        "variables": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 1], "type": "<=", "rhs": 4},  # x1 + x2 <= 4
            {"name": "c2", "coefficients": [2, 1], "type": "<=", "rhs": 5},  # 2x1 + x2 <= 5
        ],
         "bounds": {"x1": {"low": 0}, "x2": {"low": 0}} # x1 >=0, x2 >= 0 (có thể bỏ qua nếu mặc định là không âm)
    }

    sample_problem_minimize = {
        "objective": {"type": "minimize", "coefficients": [2, 5]}, # Minimize 2y1 + 5y2
        "variables": ["y1", "y2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 2], "type": ">=", "rhs": 4},  # y1 + 2y2 >= 4
            {"name": "c2", "coefficients": [3, 1], "type": ">=", "rhs": 3},  # 3y1 + y2 >= 3
        ],
        "bounds": {"y1": {"low": 0.5}} # y1 >= 0.5, y2 >= 0 (mặc định)
    }

    sample_problem_infeasible = {
        "objective": {"type": "maximize", "coefficients": [1, 1]},
        "variables": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 0], "type": ">=", "rhs": 2}, # x1 >= 2
            {"name": "c2", "coefficients": [1, 0], "type": "<=", "rhs": 1}, # x1 <= 1
        ]
    }
    
    sample_problem_unbounded = {
        "objective": {"type": "maximize", "coefficients": [1, 1]}, # Maximize x1 + x2
        "variables": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, -1], "type": ">=", "rhs": 1},  # x1 - x2 >= 1
        ],
        "bounds": {"x1": {"low": 0}, "x2": {"low": 0}}
    }


    print("--- Solving Maximization Problem ---")
    solution_max, logs_max = solve_with_pulp_cbc(sample_problem_maximize)
    for log in logs_max:
        print(log)
    if solution_max:
        print("Solution:", solution_max)

    print("\n--- Solving Minimization Problem ---")
    solution_min, logs_min = solve_with_pulp_cbc(sample_problem_minimize)
    for log in logs_min:
        print(log)
    if solution_min:
        print("Solution:", solution_min)

    print("\n--- Solving Infeasible Problem ---")
    solution_inf, logs_inf = solve_with_pulp_cbc(sample_problem_infeasible)
    for log in logs_inf:
        print(log)
    if solution_inf:
        print("Solution Status:", solution_inf.get("status"))

    print("\n--- Solving Unbounded Problem ---")
    solution_unb, logs_unb = solve_with_pulp_cbc(sample_problem_unbounded)
    for log in logs_unb:
        print(log)
    if solution_unb:
        print("Solution Status:", solution_unb.get("status"))
