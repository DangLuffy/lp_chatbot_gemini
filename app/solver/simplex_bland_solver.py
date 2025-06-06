# /app/solver/simplex_bland_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from .base_simplex_dictionary_solver import BaseSimplexDictionarySolver
from .utils import standardize_problem_for_simplex # Import hàm chuẩn hóa

logger = logging.getLogger(__name__)

class SimplexBlandSolver(BaseSimplexDictionarySolver):
    """
    Triển khai thuật toán Simplex bằng phương pháp từ điển,
    sử dụng Quy tắc Bland cho cả việc chọn biến vào và biến ra.
    Kế thừa từ BaseSimplexDictionarySolver, nơi các phương thức chọn biến
    đã mặc định theo Quy tắc Bland.
    Xử lý trường hợp RHS âm cho ràng buộc '<=' bằng một "Pha 1 đơn giản".
    Mong đợi problem_data đầu vào đã được chuẩn hóa:
    - objective: "min"
    - constraints[i].op: "<="
    """
    def __init__(self, problem_data_standardized: Dict[str, Any]):
        super().__init__(problem_data_standardized, objective_key_in_dict='z_bland') # Sử dụng key khác để phân biệt nếu cần
        self.current_phase_info: Optional[str] = None
        # BaseSimplexDictionarySolver.__init__ đã xử lý việc đọc objective, coeffs, vars
        # từ problem_data_standardized.

    def _build_initial_dictionary(self) -> bool:
        self._log("SimplexBlandSolver: Building Initial Dictionary from standardized data...")
        # self.decision_vars_names và self.objective_coeffs_list đã có từ BaseSimplex

        # 1. Xây dựng biểu thức cho hàm mục tiêu z_bland (luôn là min)
        z_expr: Dict[str, float] = {'const': 0.0}
        for i, var_name in enumerate(self.decision_vars_names):
            if i < len(self.objective_coeffs_list):
                z_expr[var_name] = self.objective_coeffs_list[i]
        self.dictionary[self.current_objective_key] = z_expr

        # 2. Xây dựng phương trình cho các ràng buộc (tất cả đã là "<=") và thêm biến bù
        constraints_from_input = self.problem_data.get("constraints", [])
        for i, constr in enumerate(constraints_from_input):
            if constr.get("op") not in ["<=", "≤"]:
                self._log(f"CRITICAL ERROR (SimplexBlandSolver): Constraint '{constr.get('name', i+1)}' received type '{constr.get('op')}' but expected '<=' after standardization.")
                return False

            slack_var_name = f"s_b{i+1}" # Tên biến bù có thể khác để tránh trùng lặp nếu có nhiều solver
            self.slack_vars_names.append(slack_var_name)
            if slack_var_name not in self.all_vars_ordered:
                self.all_vars_ordered.append(slack_var_name)

            self.basic_vars.append(slack_var_name)

            constr_expr: Dict[str, float] = {'const': constr.get("rhs", 0.0)}
            lhs_coeffs_list = constr.get("lhs", [])
            for j, var_name in enumerate(self.decision_vars_names):
                if j < len(lhs_coeffs_list):
                    constr_expr[var_name] = -lhs_coeffs_list[j]
            self.dictionary[slack_var_name] = constr_expr

        self._log(f"SimplexBlandSolver: All variables ordered after slack: {self.all_vars_ordered}")
        self._log_dictionary(phase_info="Initial Build (Standardized, Bland)")
        return True

    # _select_entering_variable và _select_leaving_variable được kế thừa từ BaseSimplexDictionarySolver,
    # chúng đã sử dụng Quy tắc Bland làm mặc định.

    def _find_leaving_var_for_phase1_simple(self) -> Optional[str]:
        """Pha 1 đơn giản: Tìm biến cơ sở có hằng số âm nhất để làm biến ra."""
        most_negative_const = -self.epsilon
        leaving_var_candidates: List[str] = []
        for var_name in self.basic_vars:
            if var_name == self.current_objective_key: continue
            const_val = self.dictionary.get(var_name, {}).get('const', 0.0)
            if const_val < most_negative_const:
                most_negative_const = const_val; leaving_var_candidates = [var_name]
            elif abs(const_val - most_negative_const) < self.epsilon and most_negative_const < -self.epsilon :
                leaving_var_candidates.append(var_name)
        if not leaving_var_candidates: return None

        # Quy tắc Bland để phá vỡ sự bằng nhau
        leaving_var_candidates.sort(key=lambda v: self.all_vars_ordered.index(v))
        leaving_var = leaving_var_candidates[0]
        self._log(f"Phase 1 (Simple - Bland) Leaving: {leaving_var} (const: {most_negative_const:.4g})")
        return leaving_var

    def _find_entering_var_for_phase1_simple(self, leaving_var: str) -> Optional[str]:
        """Pha 1 đơn giản: Tìm biến vào cho leaving_var đã chọn, sử dụng Bland."""
        leaving_var_expr = self.dictionary.get(leaving_var)
        if leaving_var_expr is None: return None
        candidate_entering_vars: List[str] = []
        # Duyệt các biến phi cơ sở theo thứ tự trong all_vars_ordered (Bland)
        sorted_non_basic = sorted(
            [nb_var for nb_var in self.non_basic_vars if nb_var in self.all_vars_ordered],
            key=lambda v_name: self.all_vars_ordered.index(v_name)
        )
        for var_name in sorted_non_basic:
            coeff_in_row = leaving_var_expr.get(var_name, 0.0)
            if coeff_in_row < -self.epsilon: # Chỉ cần hệ số âm để có thể cải thiện tính khả thi
                candidate_entering_vars.append(var_name)
        
        if not candidate_entering_vars:
            self._log(f"Phase 1 (Simple - Bland) ERROR: No suitable entering variable for {leaving_var}. Problem may be infeasible."); return None

        # Quy tắc Bland đã được áp dụng qua việc sắp xếp sorted_non_basic và chọn phần tử đầu tiên phù hợp
        # Tuy nhiên, để rõ ràng hơn, ta sẽ chọn biến có chỉ số nhỏ nhất trong số các ứng viên
        candidate_entering_vars.sort(key=lambda v: self.all_vars_ordered.index(v))
        entering_var = candidate_entering_vars[0]

        self._log(f"Phase 1 (Simple - Bland) Entering: {entering_var} (coeff in {leaving_var} row: {leaving_var_expr.get(entering_var,0.0):.4g})")
        return entering_var

    def _run_phase1_simple(self, max_phase1_iterations: int) -> str:
        """Chạy Pha 1 đơn giản để tìm một từ điển khả thi."""
        self._log("--- Starting Simple Phase 1 (Feasibility) for SimplexBlandSolver ---")
        self.current_phase_info = "Phase 1 (Feasibility - Bland)"
        phase1_iter = 0
        while phase1_iter < max_phase1_iterations:
            phase1_iter += 1; self.iteration_count +=1
            self._log_dictionary(phase_info=self.current_phase_info)
            leaving_var = self._find_leaving_var_for_phase1_simple()
            if leaving_var is None: self._log("Simple Phase 1 (Bland) completed. Dictionary is feasible."); return "Feasible"
            entering_var = self._find_entering_var_for_phase1_simple(leaving_var)
            if entering_var is None: self._log("Simple Phase 1 (Bland) FAILED. Problem likely infeasible."); return "Infeasible"
            if not self._perform_pivot(entering_var, leaving_var):
                self._log("Simple Phase 1 (Bland) FAILED: Pivot operation failed."); return "ErrorInPivot"
        self._log(f"Simple Phase 1 (Bland) FAILED: Max iterations ({max_phase1_iterations}) reached."); return "MaxIterationsReached"

    def solve(self, max_iterations: int = 50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        self.iteration_count = 0
        self._log(f"Starting SimplexBlandSolver. Max iterations: {max_iterations}")
        self._log(f"Input problem objective type (before standardization by wrapper): {self.problem_data.get('objective_type_before_standardization', 'N/A')}")
        self._log(f"Solver will use objective: {self.current_objective_type} {self.current_objective_key}")

        if not self._build_initial_dictionary():
            return self._extract_solution("ErrorInSetup"), self.logs

        initial_feasibility_check_var = self._find_leaving_var_for_phase1_simple()
        if initial_feasibility_check_var is not None:
            self._log(f"Initial dictionary not feasible (e.g., {initial_feasibility_check_var} has negative constant). Running Simple Phase 1 (Bland).")
            phase1_status = self._run_phase1_simple(max_iterations // 2 if max_iterations > 1 else 1)
            if phase1_status != "Feasible":
                return self._extract_solution(phase1_status), self.logs
        else:
            self._log("Initial dictionary is feasible. Proceeding to Optimization Phase (Bland).")

        self._log("--- Starting Optimization Phase (SimplexBlandSolver) ---")
        self.current_phase_info = "Optimization (Bland)"

        remaining_iterations = max_iterations - self.iteration_count
        if remaining_iterations <= 0: remaining_iterations = max_iterations // 2 + 1 if max_iterations > 0 else 1

        opt_phase_iter = 0
        while opt_phase_iter < remaining_iterations:
            opt_phase_iter += 1; self.iteration_count += 1
            self._log_dictionary(phase_info=self.current_phase_info)

            entering_var = self._select_entering_variable() # Kế thừa từ BaseSimplex (đã là Bland)
            if not entering_var:
                self._log("Optimization Phase (Bland): Optimal solution found.")
                return self._extract_solution("Optimal"), self.logs

            leaving_var = self._select_leaving_variable(entering_var) # Kế thừa từ BaseSimplex (đã là Bland)
            if not leaving_var:
                self._log(f"Optimization Phase (Bland): Problem is UNBOUNDED for entering var {entering_var}.")
                return self._extract_solution("Unbounded"), self.logs

            if not self._perform_pivot(entering_var, leaving_var):
                 self._log("Optimization Phase (Bland) FAILED: Pivot operation failed.")
                 return self._extract_solution("ErrorInPivot"), self.logs

        self._log(f"Max iterations ({max_iterations}) reached. Algorithm (Bland) terminated.")
        return self._extract_solution("MaxIterationsReached"), self.logs

def solve_with_simplex_bland(
    problem_data_input: Dict[str, Any], # Đây là "Định dạng A" từ DialogManager
    max_iterations=50
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Hàm bao bọc để giải bài toán LP bằng SimplexBlandSolver.
    Sẽ chuẩn hóa bài toán trước khi giải.
    """
    overall_logs: List[str] = []
    overall_logs.append("--- solve_with_simplex_bland called ---")

    standardized_problem_data, was_maximized = standardize_problem_for_simplex(problem_data_input, overall_logs)

    if standardized_problem_data is None:
        overall_logs.append("ERROR: Standardization failed for SimplexBlandSolver.")
        return {"status": "Error", "message": "Input data standardization failed."}, overall_logs
    
    # Lưu lại loại mục tiêu gốc để log
    standardized_problem_data['objective_type_before_standardization'] = problem_data_input.get("objective", "N/A")


    solver = SimplexBlandSolver(standardized_problem_data)
    solution, solver_logs = solver.solve(max_iterations=max_iterations)
    overall_logs.extend(solver_logs)

    if solution and solution.get("status") == "Optimal" and was_maximized:
        if solution.get("objective_value") is not None:
            solution["objective_value"] *= -1
            overall_logs.append(f"Final objective value (for original MAX problem, Bland) adjusted: {solution['objective_value']:.4g}")
        else:
            overall_logs.append(f"Warning (Bland): Solution status is Optimal but objective_value is None. Cannot adjust for original MAX problem.")


    elif solution and solution.get("status") == "Unbounded" and was_maximized:
        overall_logs.append("Original MAX problem is also Unbounded (Bland).")
        
    overall_logs.append("--- solve_with_simplex_bland finished ---")
    return solution, overall_logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Bài toán từ hình ảnh "Bài 3.6" (ví dụ Klee-Minty có thể gây xoay vòng cho Dantzig)
    # min 2x1 - 3x2 + 4x3
    # s.t. 2x2 + 3x3 <= 5  (s_b1)
    #      x1 + x2 + 2x3 <= 4  (s_b2)
    #      x1 + 2x2 + 3x3 <= 7  (s_b3)
    # Lời giải tối ưu: x1=0, x2=0, x3=0, z_bland=0 (vì tất cả hệ số mục tiêu là dương sau khi chuẩn hóa min, và các biến phi âm)
    # Tuy nhiên, ví dụ này đơn giản. Một ví dụ Klee-Minty thực sự sẽ phức tạp hơn.
    # Mục tiêu là kiểm tra quy tắc Bland hoạt động.
    problem_b3_6_new_format = {
        "objective": "min",
        "coeffs": [2, -3, 4], # min 2x1 -3x2 +4x3
        "variables_names_for_title_only": ["x1", "x2", "x3"],
        "constraints": [
            {"name": "w1_constr", "lhs": [0, 2, 3], "op": "<=", "rhs": 5},
            {"name": "w2_constr", "lhs": [1, 1, 2], "op": "<=", "rhs": 4},
            {"name": "w3_constr", "lhs": [1, 2, 3], "op": "<=", "rhs": 7}
            # x1,x2,x3 >= 0 được giả định bởi Simplex, không cần thêm dưới dạng ràng buộc tường minh
            # cho các solver Simplex cơ bản nếu chúng xử lý bài toán dạng chuẩn (biến không âm).
            # Nếu muốn test ràng buộc phi âm tường minh, chúng cần được thêm và chuẩn hóa thành <=.
        ]
    }

    print("--- Test 1: Solving Problem (Bài 3.6) with SimplexBlandSolver ---")
    # Với hàm mục tiêu là Min, x2 (hệ số -3) sẽ vào.
    # s_b1 = 5 - 2x2 - 3x3  => ratio x2: 5/2 = 2.5 (s_b1 ra)
    # s_b2 = 4 - x1 - x2 - 2x3 => ratio x2: 4/1 = 4
    # s_b3 = 7 - x1 - 2x2 - 3x3 => ratio x2: 7/2 = 3.5
    # s_b1 ra.
    solution1, logs1 = solve_with_simplex_bland(problem_b3_6_new_format, max_iterations=10)
    print("\n--- FULL LOGS (Test 1 - Bland) ---")
    for log_entry in logs1: print(log_entry)
    print("\n--- FINAL SOLUTION (Test 1 - Bland) ---")
    if solution1: import json; print(json.dumps(solution1, indent=2))
    # Dự kiến: x1=3, x2=1, x3=0, z_bland = 2*3 - 3*1 + 4*0 = 6 - 3 = 3. (Theo Pulp)
    # Solver tay: x2=2.5, x1=0, x3=0, z = -3*(2.5) = -7.5


    # Test với bài toán có RHS âm (Pha 1 đơn giản sẽ được kích hoạt)
    problem_b1_12_standardized_for_bland = {
        "objective": "min", # Mục tiêu đã là min
        "coeffs": [-3, -2], # Hệ số của hàm min
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            # standardize_problem_for_simplex sẽ đảm bảo tất cả là <=
            # Nên truyền vào đây dữ liệu gốc để test hàm standardize
            {"name": "RB1_orig", "lhs": [2, 1], "op": "<=", "rhs": 2},
            {"name": "RB2_orig", "lhs": [3, 4], "op": ">=", "rhs": 12} # Sẽ thành -3x1 -4x2 <= -12
        ]
    }
    print("\n\n--- Test 2: Solving Problem (Bài 1.12 - Bland) with negative RHS after standardization ---")
    solution2, logs2 = solve_with_simplex_bland(problem_b1_12_standardized_for_bland, max_iterations=10)
    print("\n--- FULL LOGS (Test 2 - Bland) ---")
    for log_entry in logs2: print(log_entry)
    print("\n--- FINAL SOLUTION (Test 2 - Bland) ---")
    if solution2: import json; print(json.dumps(solution2, indent=2))
    # Bài toán gốc: min Z = -3x1 - 2x2, s.t. 2x1+x2<=2, 3x1+4x2>=12, x1,x2>=0
    # Pulp cho nghiệm: Infeasible.

