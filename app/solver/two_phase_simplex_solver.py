# /app/solver/auxiliary_problem_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from .base_simplex_dictionary_solver import BaseSimplexDictionarySolver
from app.solver.utils import normalize_problem_data_from_nlp

logger = logging.getLogger(__name__)

class AuxiliaryProblemSolver(BaseSimplexDictionarySolver):
    """
    Triển khai phương pháp giải Simplex sử dụng bài toán bổ trợ với biến x0,
    theo cách tiếp cận được minh họa trong hình ảnh của người dùng.
    Hàm mục tiêu Pha 1 là min x0.
    Các ràng buộc đầu vào được giả định là đã ở dạng <=.
    """
    def __init__(self, problem_data: Dict[str, Any]):
        super().__init__(problem_data, objective_key='z') 
        self.auxiliary_objective_key = 'f_aux' 
        self.auxiliary_var_name = 'x0_aux'    
        self.current_phase: int = 0 

    def _build_initial_dictionary(self) -> bool:
        self._log("AuxiliaryProblemSolver: Building Initial Dictionary for Auxiliary Problem (Phase 1)...")
        
        self.dictionary = {} 
        self.basic_vars = []
        self.slack_vars_names = [] 

        # Thêm x0_aux vào danh sách biến và biến phi cơ sở ban đầu
        if self.auxiliary_var_name not in self.all_vars_ordered:
            self.all_vars_ordered.append(self.auxiliary_var_name)
        if self.auxiliary_var_name not in self.non_basic_vars:
            self.non_basic_vars.append(self.auxiliary_var_name)

        constraints = self.problem_data.get("constraints", [])
        
        if not constraints: # Nếu không có ràng buộc nào từ input
            self._log("Warning: No structural constraints provided. Phase 1: min x0_aux = 0.")
            self.current_objective_key = self.auxiliary_objective_key
            # f_aux = x0_aux. Nếu x0_aux là phi cơ sở, f_aux = 0 + 1*x0_aux
            self.dictionary[self.current_objective_key] = {'const': 0.0, self.auxiliary_var_name: 1.0} 
            self.current_objective_type = "minimize" 
            self._log_dictionary(phase_info="Phase 1 - Initial (No Constraints)")
            return True 

        # Xây dựng các dòng cho w_i từ các ràng buộc cấu trúc được cung cấp
        for i, constr_orig in enumerate(constraints):
            constr = dict(constr_orig) 
            # Giả định các ràng buộc truyền vào đây đã được chuẩn hóa thành dạng <=
            # bởi người dùng hoặc một bước tiền xử lý trước đó nếu cần.
            if constr.get("type") not in ["<=", "≤"]:
                self._log(f"ERROR (AuxiliaryProblemSolver): Constraint '{constr.get('name', i+1)}' is '{constr.get('type')}' not '<=' or '≤'. This solver's Phase 1 setup, mimicking the image, expects this form for structural constraints.")
                return False

            constr_coeffs = list(constr.get("coefficients", []))
            constr_rhs = constr.get("rhs", 0.0)
            
            slack_var_name = f"w{i+1}" 
            self.slack_vars_names.append(slack_var_name)
            if slack_var_name not in self.all_vars_ordered:
                self.all_vars_ordered.append(slack_var_name)
            self.basic_vars.append(slack_var_name) 

            # Từ điển ban đầu: w_i = b_i - sum(a_ij * x_j) + x0_aux
            row_expr: Dict[str, float] = {'const': constr_rhs}
            for j, var_name in enumerate(self.decision_vars_names):
                if j < len(constr_coeffs):
                    row_expr[var_name] = -constr_coeffs[j] 
            row_expr[self.auxiliary_var_name] = 1.0 # Luôn +x0_aux cho mỗi ràng buộc cấu trúc
            self.dictionary[slack_var_name] = row_expr
        
        # Thiết lập hàm mục tiêu Pha 1: f_aux = x0_aux (mục tiêu là min f_aux)
        self.current_objective_key = self.auxiliary_objective_key
        # Ban đầu, f_aux = 0 + 1*x0_aux (vì x0_aux là phi cơ sở)
        self.dictionary[self.current_objective_key] = {'const': 0.0, self.auxiliary_var_name: 1.0}
        self.current_objective_type = "minimize" # Mục tiêu là min x0_aux
        
        self._log(f"AuxiliaryProblemSolver: All variables ordered: {self.all_vars_ordered}")
        self._log_dictionary(phase_info="Phase 1 - Initial (before initial pivot)")

        # Thực hiện phép xoay ban đầu: x0_aux vào, w_k (có hằng số âm nhất) ra
        leaving_var_for_first_pivot: Optional[str] = None
        most_negative_const_val = self.epsilon # Chỉ tìm giá trị thực sự âm
        
        candidate_initial_leaving_vars = []
        for w_var_name in self.basic_vars: 
            if w_var_name.startswith('w'): # Đảm bảo chỉ xét các biến w_i (biến bù)
                w_expr = self.dictionary.get(w_var_name, {})
                const_in_w_row = w_expr.get('const', 0.0)
                # Chỉ xoay nếu hằng số âm VÀ dòng đó có x0_aux với hệ số 1 (để x0_aux có thể vào)
                if const_in_w_row < -self.epsilon and abs(w_expr.get(self.auxiliary_var_name, 0.0) - 1.0) < self.epsilon : 
                    candidate_initial_leaving_vars.append({'name': w_var_name, 'const': const_in_w_row})
        
        if candidate_initial_leaving_vars:
            candidate_initial_leaving_vars.sort(key=lambda x: (x['const'], self.all_vars_ordered.index(x['name'])))
            leaving_var_for_first_pivot = candidate_initial_leaving_vars[0]['name']
            most_negative_const_val = candidate_initial_leaving_vars[0]['const']
            
            self._log(f"Initial dictionary for auxiliary problem is infeasible due to {leaving_var_for_first_pivot} (const={most_negative_const_val:.4g}).")
            self._log(f"Performing initial pivot: '{self.auxiliary_var_name}' enters, '{leaving_var_for_first_pivot}' leaves.")
            
            if not self._perform_pivot(self.auxiliary_var_name, leaving_var_for_first_pivot):
                self._log("ERROR: Initial pivot for Phase 1 failed.")
                return False
            self._log_dictionary(phase_info="Phase 1 - After initial pivot")
        else:
            self._log("Initial dictionary for auxiliary problem is already feasible (all w_i consts >= 0) or no suitable initial pivot for x0_aux. x0_aux can be 0 if non-basic.")
        return True

    # _run_simplex_phase_loop, solve, _extract_solution giữ nguyên như phiên bản trước
    # vì chúng đã được thiết kế để làm việc với self.current_objective_key và self.current_objective_type
    def _run_simplex_phase_loop(self, max_phase_iterations: int) -> str:
        phase_iter = 0
        while phase_iter < max_phase_iterations:
            phase_iter += 1; self.iteration_count += 1
            self._log_dictionary(phase_info=f"Phase {self.current_phase}")
            entering_var = self._select_entering_variable() 
            if not entering_var: return "Optimal" 
            leaving_var = self._select_leaving_variable(entering_var)
            if not leaving_var: return "Unbounded"
            if not self._perform_pivot(entering_var, leaving_var): return "ErrorInPivot"
        return "MaxIterationsReached"

    def solve(self, max_iterations: int = 50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        self.iteration_count = 0
        self._log(f"Starting AuxiliaryProblemSolver. Max total iterations: {max_iterations}")

        self.current_phase = 1
        if not self._build_initial_dictionary(): 
            return self._extract_solution("ErrorInSetup"), self.logs

        phase1_status = self._run_simplex_phase_loop(max_iterations // 2 if max_iterations > 1 else 1)
        self._log(f"Phase 1 (Auxiliary Problem) ended with status: {phase1_status}")
        
        min_x0_value_at_end_phase1 = 0.0
        if self.auxiliary_var_name in self.basic_vars: # Nếu x0_aux là cơ sở
            min_x0_value_at_end_phase1 = self.dictionary.get(self.auxiliary_var_name, {}).get('const', 0.0)
        # Nếu x0_aux là phi cơ sở, giá trị của nó là 0.
        # Hoặc, nếu f_aux là mục tiêu, giá trị của f_aux là giá trị của x0
        elif self.current_objective_key == self.auxiliary_objective_key and self.auxiliary_var_name in self.dictionary.get(self.auxiliary_objective_key, {}):
             # Nếu x0_aux là phi cơ sở, và f_aux = const + 1*x0_aux + ..., thì const là giá trị của f_aux khi x0_aux=0
             # Điều này hơi phức tạp, cách đơn giản là xem giá trị của biến x0_aux
             pass # min_x0_value_at_end_phase1 đã là 0.0

        if phase1_status == "Optimal":
            if min_x0_value_at_end_phase1 > self.epsilon: 
                self._log(f"Phase 1 optimal min x0_aux = {min_x0_value_at_end_phase1:.4g} > 0. Original problem is INFEASIBLE.")
                return self._extract_solution("Infeasible"), self.logs
            
            self._log(f"Phase 1 optimal min x0_aux = {min_x0_value_at_end_phase1:.4g} approx 0. Feasible solution for original problem found.")
            if self.auxiliary_var_name in self.basic_vars: # và giá trị của nó là 0
                self._log(f"Variable {self.auxiliary_var_name} is in basis with value 0. Attempting to pivot it out.")
                x0_row = self.dictionary.get(self.auxiliary_var_name, {})
                pivot_out_candidate_for_x0: Optional[str] = None
                sorted_candidates_for_x0_pivot = sorted(
                    [nb_var for nb_var in x0_row if nb_var != 'const' and nb_var != self.auxiliary_var_name and abs(x0_row[nb_var]) > self.epsilon],
                    key=lambda v: (0 if v in self.decision_vars_names else 1, self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))
                )
                if sorted_candidates_for_x0_pivot:
                    pivot_out_candidate_for_x0 = sorted_candidates_for_x0_pivot[0]
                
                if pivot_out_candidate_for_x0:
                    self._log(f"Pivoting {self.auxiliary_var_name} out with {pivot_out_candidate_for_x0} (degenerate pivot).")
                    if not self._perform_pivot(pivot_out_candidate_for_x0, self.auxiliary_var_name):
                        self._log(f"Warning: Could not pivot out auxiliary variable {self.auxiliary_var_name}.")
                else: 
                    self._log(f"Warning: Could not find a non-auxiliary non-basic variable to pivot out {self.auxiliary_var_name}. It will be removed if non-basic.")
        elif phase1_status == "Unbounded": 
             self._log(f"Phase 1 objective {self.auxiliary_objective_key} (minimize x0_aux) is unbounded below. This means x0_aux can be arbitrarily negative, which implies original problem is feasible (x0_aux can be 0).")
             if self.auxiliary_var_name in self.basic_vars and self.dictionary.get(self.auxiliary_var_name,{}).get('const',0.0) > self.epsilon:
                self._log(f"Phase 1 unbounded, but {self.auxiliary_var_name} remains in basis > 0. This is contradictory. Assuming INFEASIBLE.")
                return self._extract_solution("Infeasible"), self.logs
        else: 
            self._log(f"Phase 1 did not complete optimally (status: {phase1_status}). Problem status uncertain.")
            return self._extract_solution(f"ErrorPhase1_{phase1_status}"), self.logs
        
        # --- CHUẨN BỊ CHO PHA 2 ---
        self._log("--- Preparing for Phase 2 ---")
        if self.auxiliary_objective_key in self.dictionary:
            del self.dictionary[self.auxiliary_objective_key]
        
        if self.auxiliary_var_name in self.dictionary: 
             del self.dictionary[self.auxiliary_var_name]
        if self.auxiliary_var_name in self.basic_vars: self.basic_vars.remove(self.auxiliary_var_name)
        
        # Loại bỏ x0_aux hoàn toàn khỏi danh sách biến phi cơ sở và all_vars_ordered cho Pha 2
        self.non_basic_vars = [v for v in self.non_basic_vars if v != self.auxiliary_var_name]
        if self.auxiliary_var_name in self.all_vars_ordered: self.all_vars_ordered.remove(self.auxiliary_var_name)

        self.current_objective_key = 'z' 
        self.current_objective_type = self.original_objective_type 
        
        z_original_expr: Dict[str, float] = {'const': 0.0}
        obj_coeffs_orig = self.problem_data.get("objective", {}).get("coefficients", [])
        for i, var_name in enumerate(self.decision_vars_names):
            if i < len(obj_coeffs_orig):
                z_original_expr[var_name] = obj_coeffs_orig[i]
        
        z_phase2_expr: Dict[str, float] = {'const': 0.0}
        for var_orig, coeff_orig in z_original_expr.items():
            if var_orig == 'const': 
                z_phase2_expr['const'] += coeff_orig; continue
            
            if var_orig in self.basic_vars: 
                basic_var_expr_for_z = self.dictionary.get(var_orig, {}) 
                z_phase2_expr['const'] += coeff_orig * basic_var_expr_for_z.get('const', 0.0)
                for nb_var, nb_coeff in basic_var_expr_for_z.items():
                    if nb_var != 'const': 
                        z_phase2_expr[nb_var] = z_phase2_expr.get(nb_var, 0.0) + coeff_orig * nb_coeff
            elif var_orig in self.non_basic_vars : 
                z_phase2_expr[var_orig] = z_phase2_expr.get(var_orig, 0.0) + coeff_orig
        
        self.dictionary['z'] = z_phase2_expr
        
        # --- PHA 2 ---
        self.current_phase = 2
        self._log("--- Starting Phase 2 Simplex Iterations ---")
        remaining_iterations = max_iterations - self.iteration_count
        if remaining_iterations <=0 : remaining_iterations = max_iterations // 2 + 1 
        
        phase2_final_status = self._run_simplex_phase_loop(max_phase_iterations=remaining_iterations)
        self._log(f"Phase 2 ended with status: {phase2_final_status}")

        return self._extract_solution(phase2_final_status), self.logs


def solve_with_auxiliary_problem_simplex(problem_data_input: Dict[str, Any], max_iterations_total=50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    overall_logs: List[str] = []
    problem_data_normalized, norm_logs = normalize_problem_data_from_nlp(problem_data_input)
    overall_logs.extend(norm_logs)

    if problem_data_normalized is None:
        overall_logs.append("ERROR: Normalization failed for AuxiliaryProblemSolver.")
        return {"status": "Error", "message": "Input data normalization failed."}, overall_logs
    
    # Chuẩn bị dữ liệu cho AuxiliaryProblemSolver:
    # Đảm bảo tất cả các ràng buộc cấu trúc là dạng <=
    # Các ràng buộc x_i >= 0 sẽ được xử lý bởi bản chất của Simplex, không cần đưa vào đây
    # nếu chúng không phải là ràng buộc cấu trúc của bài toán gốc.
    
    processed_constraints_for_aux = []
    for constr in problem_data_normalized.get("constraints", []):
        # Giả sử NLP trả về op là ">=", "<=", "=="
        # Và normalize_problem_data_from_nlp không thay đổi op
        
        # Nếu người dùng muốn x_i >= 0, họ phải nhập nó như một ràng buộc.
        # Ví dụ: {"lhs": [1,0], "op": ">=", "rhs": 0}
        # Để AuxiliaryProblemSolver này hoạt động như hình ảnh (thêm x0 vào w_i),
        # các ràng buộc này cần được chuyển thành <=.
        # -x_i <= 0
        
        # Logic này nên được thực hiện TRƯỚC KHI gọi solver,
        # hoặc solver phải đủ thông minh để xử lý các dạng op khác nhau khi thêm x0.
        # Hiện tại, _build_initial_dictionary yêu cầu op là "<=" hoặc "≤".
        processed_constraints_for_aux.append(constr)


    # Tạo problem_data mới chỉ với các ràng buộc đã xử lý (nếu cần)
    # Hoặc, sửa _build_initial_dictionary để nó tự xử lý việc chuyển đổi op
    # và thêm x0 một cách thích hợp.
    # Để đơn giản, ta sẽ giả định problem_data_normalized đã có các constraints ở dạng solver này mong đợi.

    solver = AuxiliaryProblemSolver(problem_data_normalized) # Truyền dữ liệu đã chuẩn hóa
    solution, solver_logs = solver.solve(max_iterations=max_iterations_total) 
    overall_logs.extend(solver_logs)
    return solution, overall_logs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Bài toán từ hình ảnh image_f72cfa.jpg (Bài 1.20 trong hình đó)
    # min x1 + x2
    # Ràng buộc gốc:
    # -x1 + x2 <= -2  (1)
    #  x1 + 2x2 <= 4  (2)
    #  x1       <= 1  (3)
    # x1, x2 >= 0 (Trong hình, các ràng buộc này không được thêm x0 vào w_i tương ứng)
    # Để mô phỏng, ta sẽ chỉ đưa 3 ràng buộc cấu trúc vào problem_data_input.
    # Hàm normalize_problem_data_from_nlp sẽ tạo tên biến x1, x2.
    # AuxiliaryProblemSolver sẽ không tự thêm ràng buộc x1,x2 >=0.
    # Chúng được giả định bởi Simplex.

    problem_f72cfa_input = {
        "objective": "min", 
        "coeffs": [1, 1], 
        "variables_names_for_title_only": ["x1", "x2"], 
        "constraints": [
            # Các ràng buộc này đều ở dạng <=, sẽ được AuxiliaryProblemSolver xử lý
            {"name": "R1", "lhs": [-1, 1], "op": "<=", "rhs": -2},
            {"name": "R2", "lhs": [1, 2], "op": "<=", "rhs": 4},
            {"name": "R3", "lhs": [1, 0], "op": "<=", "rhs": 1}
            # tự động giả sử x1>=0, x2>=0 ở đây.
        ]
    }
    print("--- Solving Problem from image_f72cfa.jpg with AuxiliaryProblemSolver ---")
    solution_f72, logs_f72 = solve_with_auxiliary_problem_simplex(problem_f72cfa_input, max_iterations_total=20)
    
    print("\n--- FULL LOGS (image_f72cfa.jpg) ---")
    for log_entry in logs_f72: print(log_entry)
    print("\n--- FINAL SOLUTION (image_f72cfa.jpg) ---")
    if solution_f72: import json; print(json.dumps(solution_f72, indent=2))
    # Expected: Infeasible, vì x0_min sẽ là 1/2 > 0 theo các bước trong hình.
