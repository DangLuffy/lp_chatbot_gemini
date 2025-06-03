# /app/solver/simple_dictionary_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from .base_simplex_dictionary_solver import BaseSimplexDictionarySolver
from app.solver.utils import normalize_problem_data_from_nlp

logger = logging.getLogger(__name__)

class SimpleDictionarySolver(BaseSimplexDictionarySolver):
    """
    Triển khai thuật toán Simplex bằng phương pháp từ điển,
    sử dụng quy tắc Dantzig để chọn biến vào.
    Kế thừa từ BaseSimplexDictionarySolver.
    Xử lý trường hợp RHS âm cho ràng buộc '<=' bằng một "Pha 1 đơn giản".
    """
    def __init__(self, problem_data: Dict[str, Any]):
        super().__init__(problem_data, objective_key='z')
        self.current_phase_info: Optional[str] = None

    def _build_initial_dictionary(self) -> bool:
        self._log("SimpleDictionarySolver: Building Initial Dictionary...")
        # self._build_initial_dictionary_common_setup() đã được gọi trong __init__ của lớp cha

        z_expr: Dict[str, float] = {'const': 0.0}
        obj_coeffs_list = self.problem_data.get("objective", {}).get("coefficients", [])
        for i, var_name in enumerate(self.decision_vars_names):
            if i < len(obj_coeffs_list):
                z_expr[var_name] = obj_coeffs_list[i]
        self.dictionary[self.current_objective_key] = z_expr

        constraints = self.problem_data.get("constraints", [])
        for i, constr in enumerate(constraints):
            if constr.get("type") not in ["<=", "≤"]:
                self._log(f"ERROR (SimpleDictionarySolver): Constraint '{constr.get('name', i+1)}' is not '<=' or '≤'. This solver expects constraints standardized to '<=' for simple dictionary setup.")
                return False
            
            slack_var_name = f"s{i+1}"
            self.slack_vars_names.append(slack_var_name)
            if slack_var_name not in self.all_vars_ordered:
                self.all_vars_ordered.append(slack_var_name) 
            
            self.basic_vars.append(slack_var_name)

            constr_expr: Dict[str, float] = {'const': constr.get("rhs", 0.0)}
            constr_coeffs_list = constr.get("coefficients", [])
            for j, var_name in enumerate(self.decision_vars_names):
                if j < len(constr_coeffs_list):
                    constr_expr[var_name] = -constr_coeffs_list[j] 
            self.dictionary[slack_var_name] = constr_expr
        
        self._log(f"SimpleDictionarySolver: All variables ordered: {self.all_vars_ordered}")
        self._log_dictionary(phase_info="Initial Build") 
        return True

    def _select_entering_variable(self) -> Optional[str]:
        """Chọn biến vào cơ sở theo Quy tắc Dantzig."""
        obj_expr = self.dictionary.get(self.current_objective_key)
        if obj_expr is None:
            self._log(f"ERROR: Objective key '{self.current_objective_key}' not found for entering var selection.")
            return None
            
        best_coeff_improvement = -self.epsilon # Cho max, tìm dương lớn nhất. Cho min, tìm âm nhỏ nhất (giá trị tuyệt đối lớn nhất).
        entering_var: Optional[str] = None
        
        # Sắp xếp các biến phi cơ sở theo chỉ số để đảm bảo tính nhất quán nếu có nhiều biến cùng cải thiện tốt nhất
        # (Mặc dù Dantzig không yêu cầu, nhưng làm vậy giúp kết quả dễ đoán hơn)
        sorted_non_basic_vars = sorted(self.non_basic_vars, key=lambda v_name: self.all_vars_ordered.index(v_name))

        if self.current_objective_type == "maximize":
            for var_name in sorted_non_basic_vars:
                coeff = obj_expr.get(var_name, 0.0)
                if coeff > best_coeff_improvement + self.epsilon : # Tìm hệ số dương lớn nhất
                    best_coeff_improvement = coeff
                    entering_var = var_name
        else: # current_objective_type == "minimize"
            # Đối với minimize, z = const + c1x1 + c2x2... Ta muốn tăng x_j nếu c_j < 0.
            # Chọn c_j âm nhất (có giá trị tuyệt đối lớn nhất).
            # best_coeff_improvement khởi tạo là epsilon (một số dương nhỏ)
            best_coeff_improvement = self.epsilon 
            for var_name in sorted_non_basic_vars:
                coeff = obj_expr.get(var_name, 0.0)
                if coeff < best_coeff_improvement - self.epsilon: # Tìm hệ số âm "âm nhất"
                    best_coeff_improvement = coeff
                    entering_var = var_name
        
        if entering_var is None:
            self._log(f"Optimality condition met for {self.current_objective_key}. No candidates for entering variable (Dantzig).")
            return None
        
        self._log(f"Selected Entering (Dantzig): {entering_var} (coeff in {self.current_objective_key}: {best_coeff_improvement:.4g}, index: {self.all_vars_ordered.index(entering_var)})")
        return entering_var

    # _select_leaving_variable sẽ được kế thừa từ BaseSimplexDictionarySolver (sử dụng Bland's tie-breaker)

    def _find_leaving_var_for_phase1_simple(self) -> Optional[str]:
        """Pha 1 đơn giản: Tìm biến cơ sở có hằng số âm nhất để làm biến ra."""
        # (Giữ nguyên logic từ SimplexBlandSolver)
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
        leaving_var_candidates.sort(key=lambda v: self.all_vars_ordered.index(v))
        leaving_var = leaving_var_candidates[0]
        self._log(f"Phase 1 (Simple) Leaving (Bland for tie-break): {leaving_var} (const: {most_negative_const:.4g})")
        return leaving_var

    def _find_entering_var_for_phase1_simple(self, leaving_var: str) -> Optional[str]:
        """Pha 1 đơn giản: Tìm biến vào cho leaving_var đã chọn.
        Chọn biến phi cơ sở x_j có hệ số âm trong dòng của leaving_var (trong từ điển).
        Theo quy tắc Bland (chỉ số nhỏ nhất) để đảm bảo kết thúc.
        """
        # (Giữ nguyên logic từ SimplexBlandSolver)
        leaving_var_expr = self.dictionary.get(leaving_var)
        if leaving_var_expr is None: return None
        candidate_entering_vars: List[str] = []
        for var_name in self.non_basic_vars:
            coeff_in_row = leaving_var_expr.get(var_name, 0.0)
            if coeff_in_row < -self.epsilon: candidate_entering_vars.append(var_name)
        if not candidate_entering_vars:
            self._log(f"Phase 1 (Simple) ERROR: No suitable entering variable for {leaving_var}. Problem may be infeasible."); return None 
        candidate_entering_vars.sort(key=lambda v: self.all_vars_ordered.index(v))
        entering_var = candidate_entering_vars[0]
        self._log(f"Phase 1 (Simple) Entering (Bland): {entering_var} (coeff in {leaving_var} row: {leaving_var_expr.get(entering_var,0.0):.4g})")
        return entering_var

    def _run_phase1_simple(self, max_phase1_iterations: int) -> str:
        """Chạy Pha 1 đơn giản để tìm một từ điển khả thi."""
        # (Giữ nguyên logic từ SimplexBlandSolver)
        self._log("--- Starting Simple Phase 1 (Feasibility) for SimpleDictionarySolver ---")
        self.current_phase_info = "Phase 1 (Feasibility)"
        phase1_iter = 0
        while phase1_iter < max_phase1_iterations:
            phase1_iter += 1; self.iteration_count +=1
            self._log_dictionary(phase_info=self.current_phase_info)
            leaving_var = self._find_leaving_var_for_phase1_simple()
            if leaving_var is None: self._log("Simple Phase 1 completed. Dictionary is feasible."); return "Feasible" 
            entering_var = self._find_entering_var_for_phase1_simple(leaving_var)
            if entering_var is None: self._log("Simple Phase 1 FAILED. Problem likely infeasible."); return "Infeasible" 
            if not self._perform_pivot(entering_var, leaving_var):
                self._log("Simple Phase 1 FAILED: Pivot operation failed."); return "ErrorInPivot"
        self._log(f"Simple Phase 1 FAILED: Max iterations ({max_phase1_iterations}) reached."); return "MaxIterationsReached"


    def solve(self, max_iterations: int = 50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        self.iteration_count = 0
        self._log(f"Starting SimpleDictionarySolver (Dantzig for entering). Max iterations: {max_iterations}")

        if not self._build_initial_dictionary():
            return self._extract_solution("ErrorInSetup"), self.logs

        initial_feasibility_check_var = self._find_leaving_var_for_phase1_simple()
        if initial_feasibility_check_var is not None:
            self._log(f"Initial dictionary not feasible. Running Simple Phase 1.")
            phase1_status = self._run_phase1_simple(max_iterations // 2)
            if phase1_status != "Feasible":
                return self._extract_solution(phase1_status), self.logs
        else:
            self._log("Initial dictionary is feasible. Proceeding to Optimization Phase.")

        self._log("--- Starting Optimization Phase (SimpleDictionarySolver) ---")
        self.current_phase_info = "Optimization"
        
        remaining_iterations = max_iterations - self.iteration_count 
        opt_phase_iter = 0

        while opt_phase_iter < remaining_iterations:
            opt_phase_iter += 1; self.iteration_count += 1
            self._log_dictionary(phase_info=self.current_phase_info)
            
            entering_var = self._select_entering_variable() # Sử dụng Dantzig
            if not entering_var: 
                self._log("Optimization Phase: Optimal solution found.")
                return self._extract_solution("Optimal"), self.logs
            
            leaving_var = self._select_leaving_variable(entering_var) # Sử dụng Bland tie-breaker từ lớp cha
            if not leaving_var: 
                self._log(f"Optimization Phase: Problem is UNBOUNDED for entering var {entering_var}.")
                return self._extract_solution("Unbounded"), self.logs
                
            if not self._perform_pivot(entering_var, leaving_var):
                 self._log("Optimization Phase FAILED: Pivot operation failed.")
                 return self._extract_solution("ErrorInPivot"), self.logs
            
        self._log(f"Max iterations ({max_iterations}) reached. Algorithm terminated.")
        return self._extract_solution("MaxIterationsReached"), self.logs

def solve_with_simple_dictionary(problem_data_input: Dict[str, Any], max_iterations=50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    overall_logs: List[str] = []
    problem_data_normalized, norm_logs = normalize_problem_data_from_nlp(problem_data_input)
    overall_logs.extend(norm_logs)

    if problem_data_normalized is None:
        overall_logs.append("ERROR: Normalization failed for SimpleDictionarySolver.")
        return {"status": "Error", "message": "Input data normalization failed."}, overall_logs
    
    solver = SimpleDictionarySolver(problem_data_normalized)
    solution, solver_logs = solver.solve(max_iterations=max_iterations)
    overall_logs.extend(solver_logs)
    
    return solution, overall_logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Bài toán từ hình ảnh "Bài 3.6" - để so sánh với Bland
    problem_b3_6_new_format = {
        "objective": "min", 
        "coeffs": [2, -3, 4], 
        "variables_names_for_title_only": ["x1", "x2", "x3"], 
        "constraints": [
            {"name": "w1_constr", "lhs": [0, 2, 3], "op": "<=", "rhs": 5}, 
            {"name": "w2_constr", "lhs": [1, 1, 2], "op": "<=", "rhs": 4}, 
            {"name": "w3_constr", "lhs": [1, 2, 3], "op": "<=", "rhs": 7}  
        ]
    }

    print("--- Solving Problem (Bài 3.6) with SimpleDictionarySolver (Dantzig) ---")
    solution_b3_6, logs_b3_6 = solve_with_simple_dictionary(problem_b3_6_new_format, max_iterations=10)
    
    print("\n--- FULL LOGS (Bài 3.6 - SimpleDictionarySolver) ---")
    for log_entry in logs_b3_6: print(log_entry)
    print("\n--- FINAL SOLUTION (Bài 3.6 - SimpleDictionarySolver) ---")
    if solution_b3_6: import json; print(json.dumps(solution_b3_6, indent=2))
    else: print("No solution found or an error occurred.")

    # Bài toán từ hình ảnh "Bài 1.12 - trang 42" (image_9055fc.jpg)
    # "Dạng chuẩn" (P') trong hình: min -3x1 - 2x2
    # 2x1 + x2 <= 2      (1')
    # -3x1 - 4x2 <= -12  (2') <--- RHS âm, Pha 1 đơn giản sẽ xử lý
    print("\n\n--- Solving Problem (Bài 1.12) with SimpleDictionarySolver ---")
    problem_b1_12_new_format = {
        "objective": "min", 
        "coeffs": [-3, -2],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "RB1_prime", "lhs": [2, 1], "op": "<=", "rhs": 2},
            {"name": "RB2_prime", "lhs": [-3, -4], "op": "<=", "rhs": -12},
            {"name": "RB3_prime", "lhs": [1, 0], "op": ">=", "rhs": 0}  # Thêm ràng buộc phi âm cho x1
            {"name": "RB4_prime", "lhs": [0, 1], "op": ">=", "rhs": 0}  # Thêm ràng buộc phi âm cho x2
        ]
    }
    solution_b1_12, logs_b1_12 = solve_with_simple_dictionary(problem_b1_12_new_format, max_iterations=10)
    print("\n--- FULL LOGS (Bài 1.12 - SimpleDictionarySolver) ---")
    for log_entry in logs_b1_12: print(log_entry)
    print("\n--- FINAL SOLUTION (Bài 1.12 - SimpleDictionarySolver) ---")
    if solution_b1_12: import json; print(json.dumps(solution_b1_12, indent=2))

