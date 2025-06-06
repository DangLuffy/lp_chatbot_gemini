# /app/solver/auxiliary_problem_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from .base_simplex_dictionary_solver import BaseSimplexDictionarySolver
from .utils import standardize_problem_for_simplex

logger = logging.getLogger(__name__)

class AuxiliaryProblemSolver(BaseSimplexDictionarySolver):
    """
    Triển khai phương pháp giải Simplex sử dụng bài toán bổ trợ với biến x0_aux,
    theo cách tiếp cận được minh họa trong hình ảnh của người dùng (min x0).
    Mong đợi problem_data đầu vào đã được chuẩn hóa bởi hàm bao bọc:
    - objective: "min" (đây là hàm mục tiêu gốc, đã được chuẩn hóa)
    - constraints[i].op: "<=" (tất cả ràng buộc cấu trúc đã được chuẩn hóa)
    """
    def __init__(self, problem_data_standardized: Dict[str, Any]):
        # problem_data_standardized chứa hàm mục tiêu GỐC (đã được chuẩn hóa thành min nếu cần)
        # và các ràng buộc GỐC (đã được chuẩn hóa thành <=)
        # objective_key_in_dict cho BaseSimplex là cho hàm mục tiêu GỐC (Pha 2)
        super().__init__(problem_data_standardized, objective_key_in_dict='z_original')
        
        self.auxiliary_objective_key = 'f_aux'  # Hàm mục tiêu cho Pha 1: min f_aux (f_aux = x0_aux)
        self.auxiliary_var_name = 'x0_aux'
        self.current_phase: int = 0
        self.original_problem_objective_coeffs: List[float] = list(self.problem_data.get("coeffs", [])) # Lưu hệ số mục tiêu gốc (đã chuẩn hóa min)
        self.phase1_solution_status: Optional[str] = None

    def _build_initial_dictionary(self) -> bool:
        self._log("AuxiliaryProblemSolver: Building Initial Dictionary for Auxiliary Problem (Phase 1)...")
        self.current_phase = 1
        self.current_objective_key = self.auxiliary_objective_key # Chuyển sang hàm mục tiêu Pha 1
        self.current_objective_type = "minimize" # Pha 1 luôn là minimize x0_aux

        self.dictionary = {}
        self.basic_vars = [] # Sẽ chứa các w_i ban đầu
        self.slack_vars_names = [] # Đổi tên từ slack_vars_names của BaseSimplex, đây là các w_i

        # Thêm auxiliary_var_name (x0_aux) vào danh sách biến và biến phi cơ sở ban đầu
        if self.auxiliary_var_name not in self.all_vars_ordered:
            self.all_vars_ordered.append(self.auxiliary_var_name)
        
        # Đảm bảo x0_aux là phi cơ sở khi bắt đầu Pha 1
        if self.auxiliary_var_name not in self.non_basic_vars:
             if self.auxiliary_var_name in self.basic_vars: # Nếu lỡ có trong basic
                 self.basic_vars.remove(self.auxiliary_var_name)
             self.non_basic_vars.append(self.auxiliary_var_name)
        # Sắp xếp lại non_basic_vars
        self.non_basic_vars.sort(key=lambda v: self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))


        # self.problem_data chứa các ràng buộc đã được chuẩn hóa thành "<="
        constraints_standardized = self.problem_data.get("constraints", [])

        if not constraints_standardized:
            self._log("Warning: No structural constraints provided for AuxiliaryProblemSolver. Phase 1: min x0_aux = 0.")
            # f_aux = x0_aux. Nếu x0_aux là phi cơ sở, f_aux = 0 + 1*x0_aux
            self.dictionary[self.auxiliary_objective_key] = {'const': 0.0, self.auxiliary_var_name: 1.0}
            self._log_dictionary(phase_info="Phase 1 - Initial (No Constraints)")
            return True

        # Xây dựng các dòng cho w_i từ các ràng buộc cấu trúc (đã là "<=")
        for i, constr in enumerate(constraints_standardized):
            if constr.get("op") not in ["<=", "≤"]:
                self._log(f"CRITICAL ERROR (AuxProblemSolver): Constraint '{constr.get('name', i+1)}' type is '{constr.get('op')}' but expected '<=' after pre-standardization.")
                return False

            # Đặt tên biến bù/slack cho ràng buộc gốc này là w_i
            # Đây là các biến sẽ được thêm x0_aux vào
            slack_var_for_aux_name = f"w{i+1}_aux" # Đặt tên riêng để không lẫn với slack_vars_names của Base
            self.slack_vars_names.append(slack_var_for_aux_name) # Lưu tên các biến w_i này
            if slack_var_for_aux_name not in self.all_vars_ordered:
                self.all_vars_ordered.append(slack_var_for_aux_name)
            self.basic_vars.append(slack_var_for_aux_name) # w_i là biến cơ sở ban đầu

            # Từ điển ban đầu: w_i = rhs - sum(lhs_coeffs * decision_vars) + x0_aux
            row_expr: Dict[str, float] = {'const': constr.get("rhs", 0.0)}
            lhs_coeffs_list = constr.get("lhs", [])
            for j, var_name in enumerate(self.decision_vars_names): # decision_vars_names từ BaseSimplex
                if j < len(lhs_coeffs_list):
                    row_expr[var_name] = -lhs_coeffs_list[j]
            
            row_expr[self.auxiliary_var_name] = 1.0 # Luôn +1*x0_aux cho mỗi ràng buộc cấu trúc
            self.dictionary[slack_var_for_aux_name] = row_expr

        # Thiết lập hàm mục tiêu Pha 1: f_aux = x0_aux (mục tiêu là min f_aux)
        # Ban đầu, x0_aux là phi cơ sở, nên f_aux = 0 + 1*x0_aux
        self.dictionary[self.auxiliary_objective_key] = {'const': 0.0, self.auxiliary_var_name: 1.0}

        self._log(f"AuxiliaryProblemSolver: All variables ordered: {self.all_vars_ordered}")
        self._log_dictionary(phase_info="Phase 1 - Initial (Before Initial Pivot)")

        # Thực hiện phép xoay ban đầu: x0_aux vào, w_k (có hằng số âm nhất) ra
        # để làm cho từ điển Pha 1 khả thi (tất cả hằng số của w_i >= 0)
        leaving_var_for_first_pivot: Optional[str] = None
        most_negative_const_val = self.epsilon # Chỉ tìm giá trị thực sự âm

        candidate_initial_leaving_vars = []
        # Sắp xếp các biến w_i theo chỉ số để chọn nhất quán
        sorted_w_vars = sorted(
            [w_var for w_var in self.basic_vars if w_var.startswith('w') and w_var.endswith('_aux')],
            key=lambda v_name: self.all_vars_ordered.index(v_name)
        )

        for w_var_name in sorted_w_vars:
            w_expr = self.dictionary.get(w_var_name, {})
            const_in_w_row = w_expr.get('const', 0.0)
            # Chỉ xoay nếu hằng số âm VÀ dòng đó có x0_aux với hệ số 1 (để x0_aux có thể vào)
            if const_in_w_row < -self.epsilon and abs(w_expr.get(self.auxiliary_var_name, 0.0) - 1.0) < self.epsilon:
                candidate_initial_leaving_vars.append({'name': w_var_name, 'const': const_in_w_row})

        if candidate_initial_leaving_vars:
            # Chọn w_k có hằng số âm nhất, nếu bằng nhau thì theo chỉ số (Bland)
            candidate_initial_leaving_vars.sort(key=lambda x: (x['const'], self.all_vars_ordered.index(x['name'])))
            leaving_var_for_first_pivot = candidate_initial_leaving_vars[0]['name']
            most_negative_const_val = candidate_initial_leaving_vars[0]['const']

            self._log(f"Initial dictionary for auxiliary problem is infeasible due to {leaving_var_for_first_pivot} (const={most_negative_const_val:.4g}).")
            self._log(f"Performing initial pivot: '{self.auxiliary_var_name}' enters, '{leaving_var_for_first_pivot}' leaves.")

            if not self._perform_pivot(self.auxiliary_var_name, leaving_var_for_first_pivot):
                self._log("ERROR: Initial pivot for Phase 1 failed.")
                return False
            self._log_dictionary(phase_info="Phase 1 - After Initial Pivot")
        else:
            self._log("Initial dictionary for auxiliary problem is already feasible (all w_i consts >= 0 or no suitable initial pivot for x0_aux). x0_aux can be 0 if non-basic.")
        
        # Sắp xếp lại all_vars_ordered sau khi thêm các biến w_i và x0_aux
        self.all_vars_ordered.sort(key=lambda v: (
            0 if v in self.decision_vars_names else 1 if v.startswith('s') and '_b' not in v and '_aux' not in v else 2 if v.startswith('w') and v.endswith('_aux') else 3 if v == self.auxiliary_var_name else 4,
            v # Sắp xếp theo tên nếu cùng loại
        ))
        return True

    def _run_simplex_phase_loop(self, max_phase_iterations: int) -> str:
        """Chạy vòng lặp Simplex cho pha hiện tại (Pha 1 hoặc Pha 2)."""
        phase_iter = 0
        while phase_iter < max_phase_iterations:
            phase_iter += 1
            self.iteration_count += 1 # Tăng tổng số vòng lặp
            self._log_dictionary(phase_info=f"Phase {self.current_phase} - Iter {phase_iter}")
            
            # _select_entering_variable và _select_leaving_variable được kế thừa từ BaseSimplex
            # và sử dụng self.current_objective_key (sẽ là f_aux cho Pha 1, z_original cho Pha 2)
            # và self.current_objective_type (luôn là "minimize" cho cả hai pha sau chuẩn hóa)
            entering_var = self._select_entering_variable()
            if not entering_var:
                return "Optimal" # Tối ưu cho pha hiện tại

            leaving_var = self._select_leaving_variable(entering_var)
            if not leaving_var:
                # Đối với Pha 1 (min x0_aux), nếu không bị chặn nghĩa là x0_aux có thể tiến tới -vô cùng.
                # Điều này có nghĩa là x0_aux có thể bằng 0, tức là bài toán gốc khả thi.
                if self.current_phase == 1:
                    self._log(f"Phase 1 (min {self.auxiliary_var_name}) is unbounded below. This implies the original problem is feasible.")
                    # Ta cần đảm bảo x0_aux thực sự có thể bằng 0.
                    # Nếu x0_aux vào cơ sở với giá trị 0, coi như tối ưu cho Pha 1 với giá trị 0.
                    # Hoặc nếu f_aux không bị chặn nhưng x0_aux có thể được đẩy ra khỏi cơ sở ở giá trị 0.
                    # Đây là trường hợp phức tạp, thường thì "Unbounded" cho min x0 có nghĩa là x0 có thể < 0.
                    # Nếu mục tiêu là min x0 >= 0, thì "Unbounded" sẽ không xảy ra nếu có nghiệm.
                    # Với cách thiết lập f_aux = x0_aux, nếu f_aux unbounded -> x0_aux unbounded.
                    # Nếu x0_aux có thể < 0 tùy ý, thì nó cũng có thể = 0.
                    return "Optimal" # Coi như tối ưu với x0=0 nếu Pha 1 không bị chặn
                else: # Pha 2
                    return "Unbounded" # Bài toán gốc không bị chặn

            if not self._perform_pivot(entering_var, leaving_var):
                return "ErrorInPivot"
        return "MaxIterationsReached"

    def solve(self, max_iterations: int = 50) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        self.iteration_count = 0
        self._log(f"Starting AuxiliaryProblemSolver. Max total iterations: {max_iterations}")
        self._log(f"Input problem objective type (before any standardization by wrapper): {self.problem_data.get('objective_type_before_standardization', 'N/A')}")


        # --- PHA 1: Giải bài toán phụ trợ min f_aux = x0_aux ---
        if not self._build_initial_dictionary(): # _build_initial_dictionary đã đặt current_phase = 1
            return self._extract_solution("ErrorInSetup"), self.logs

        max_phase1_iters = max_iterations // 2 if max_iterations > 1 else 1
        self.phase1_solution_status = self._run_simplex_phase_loop(max_phase1_iters)
        self._log(f"Phase 1 (Auxiliary Problem min {self.auxiliary_var_name}) ended with status: {self.phase1_solution_status}")

        # Kiểm tra giá trị tối ưu của x0_aux
        final_aux_obj_value = self.dictionary.get(self.auxiliary_objective_key, {}).get('const', 0.0)
        
        # Cũng kiểm tra giá trị của x0_aux nếu nó là biến cơ sở
        x0_aux_value_if_basic = 0.0
        if self.auxiliary_var_name in self.basic_vars:
            x0_aux_value_if_basic = self.dictionary.get(self.auxiliary_var_name,{}).get('const', 0.0)
        
        self._log(f"Value of auxiliary objective '{self.auxiliary_objective_key}' at end of Phase 1: {final_aux_obj_value:.4g}")
        if self.auxiliary_var_name in self.basic_vars:
             self._log(f"Value of basic variable '{self.auxiliary_var_name}' at end of Phase 1: {x0_aux_value_if_basic:.4g}")


        if self.phase1_solution_status == "Optimal":
            # Nếu giá trị tối ưu của f_aux (tức là x0_aux) > epsilon (dương đáng kể)
            # Hoặc nếu x0_aux là biến cơ sở và giá trị của nó > epsilon
            if final_aux_obj_value > self.epsilon or (self.auxiliary_var_name in self.basic_vars and x0_aux_value_if_basic > self.epsilon):
                self._log(f"Phase 1 optimal min {self.auxiliary_var_name} = {max(final_aux_obj_value, x0_aux_value_if_basic):.4g} > 0. Original problem is INFEASIBLE.")
                return self._extract_solution("Infeasible"), self.logs
            else: # Giá trị tối ưu của x0_aux xấp xỉ 0
                self._log(f"Phase 1 optimal min {self.auxiliary_var_name} approx 0. Feasible solution for original problem may exist.")
                # Nếu x0_aux vẫn còn trong cơ sở với giá trị 0, cố gắng đẩy nó ra
                if self.auxiliary_var_name in self.basic_vars and abs(x0_aux_value_if_basic) < self.epsilon:
                    self._log(f"Variable {self.auxiliary_var_name} is in basis with value ~0. Attempting to pivot it out.")
                    x0_row = self.dictionary.get(self.auxiliary_var_name, {})
                    pivot_out_candidate_for_x0: Optional[str] = None
                    # Ưu tiên đưa biến quyết định vào để thay thế x0_aux
                    sorted_candidates_for_x0_pivot = sorted(
                        [nb_var for nb_var in x0_row if nb_var != 'const' and nb_var != self.auxiliary_var_name and abs(x0_row[nb_var]) > self.epsilon and nb_var in self.non_basic_vars],
                        key=lambda v: (0 if v in self.decision_vars_names else 1, self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))
                    )
                    if sorted_candidates_for_x0_pivot:
                        pivot_out_candidate_for_x0 = sorted_candidates_for_x0_pivot[0]

                    if pivot_out_candidate_for_x0:
                        self._log(f"Pivoting {self.auxiliary_var_name} out with {pivot_out_candidate_for_x0} (degenerate pivot).")
                        if not self._perform_pivot(pivot_out_candidate_for_x0, self.auxiliary_var_name):
                            self._log(f"Warning: Could not pivot out auxiliary variable {self.auxiliary_var_name}. It will be removed if non-basic.")
                            # Nếu không xoay ra được, và nó là phi cơ sở (sau khi thử xoay), thì vẫn OK
                            if self.auxiliary_var_name in self.non_basic_vars:
                                self._log(f"{self.auxiliary_var_name} became non-basic after failed pivot attempt or was already non-basic.")
                            else: # Vẫn là cơ sở, đây là vấn đề
                                 self._log(f"ERROR: {self.auxiliary_var_name} is still basic after failed pivot. Problematic for Phase 2.")
                                 return self._extract_solution("ErrorPhase1_AuxVarInBasis"), self.logs

                    else:
                        self._log(f"Warning: Could not find a non-basic variable to pivot out {self.auxiliary_var_name}. It will be removed if non-basic. If it remains basic with value 0 and all its coeffs for non-basics are 0, this row might be redundant.")
                        # Nếu x0_aux là cơ sở, giá trị 0, và dòng của nó chỉ có hằng số 0 (các hệ số khác của biến phi cơ sở là 0)
                        # thì dòng đó có thể là dư thừa.
                        is_redundant_x0_row = True
                        if self.auxiliary_var_name in self.basic_vars:
                             x0_expr_check = self.dictionary.get(self.auxiliary_var_name, {})
                             if abs(x0_expr_check.get('const',0.0)) > self.epsilon: is_redundant_x0_row = False
                             for nb_v_check in self.non_basic_vars:
                                 if abs(x0_expr_check.get(nb_v_check,0.0)) > self.epsilon:
                                     is_redundant_x0_row = False; break
                             if is_redundant_x0_row:
                                 self._log(f"Row for basic {self.auxiliary_var_name} is redundant (0 = 0). Removing it for Phase 2.")
                                 self.basic_vars.remove(self.auxiliary_var_name)
                                 # self.all_vars_ordered.remove(self.auxiliary_var_name) # Không xóa khỏi all_vars vội, chỉ là không còn cơ sở
                                 # del self.dictionary[self.auxiliary_var_name] # Dòng này sẽ bị xóa khi chuẩn bị Pha 2
                        if not is_redundant_x0_row and self.auxiliary_var_name in self.basic_vars:
                             self._log(f"ERROR: {self.auxiliary_var_name} is basic, non-zero, or non-redundant. Problematic for Phase 2.")
                             return self._extract_solution("ErrorPhase1_AuxVarInBasis"), self.logs

        elif self.phase1_solution_status == "Unbounded":
             # Nếu min x0_aux không bị chặn dưới, nghĩa là x0_aux có thể < 0 tùy ý.
             # Điều này có nghĩa là luôn có thể tìm được x0_aux = 0 (và < 0).
             # Do đó, bài toán gốc là khả thi.
             self._log(f"Phase 1 objective (min {self.auxiliary_var_name}) is unbounded below. Original problem is FEASIBLE.")
             # Cần đảm bảo x0_aux không còn trong cơ sở hoặc có giá trị 0 nếu trong cơ sở.
             # Nếu x0_aux là cơ sở và có giá trị > 0, đó là mâu thuẫn.
             if self.auxiliary_var_name in self.basic_vars and x0_aux_value_if_basic > self.epsilon:
                self._log(f"Phase 1 unbounded, but {self.auxiliary_var_name} remains in basis > 0. This is contradictory or indicates an issue. Assuming INFEASIBLE for safety.")
                return self._extract_solution("Infeasible"), self.logs
        else: # Các trạng thái lỗi khác từ Pha 1
            self._log(f"Phase 1 did not complete optimally (status: {self.phase1_solution_status}). Original problem status uncertain.")
            return self._extract_solution(f"ErrorPhase1_{self.phase1_solution_status}"), self.logs

        # --- CHUẨN BỊ CHO PHA 2 ---
        self._log("--- Preparing for Phase 2 (Original Problem) ---")
        # 1. Xóa hàm mục tiêu phụ trợ f_aux
        if self.auxiliary_objective_key in self.dictionary:
            del self.dictionary[self.auxiliary_objective_key]
            self._log(f"Removed auxiliary objective '{self.auxiliary_objective_key}' from dictionary.")

        # 2. Xử lý biến phụ trợ x0_aux
        # Nếu x0_aux là phi cơ sở, nó sẽ có giá trị 0 trong các dòng khác.
        # Chúng ta cần xóa cột của x0_aux khỏi tất cả các biểu thức.
        if self.auxiliary_var_name in self.non_basic_vars:
            self._log(f"Auxiliary variable '{self.auxiliary_var_name}' is non-basic. Removing its column from all dictionary rows.")
            for basic_var_expr_key in list(self.dictionary.keys()):
                if self.auxiliary_var_name in self.dictionary[basic_var_expr_key]:
                    del self.dictionary[basic_var_expr_key][self.auxiliary_var_name]
            # Xóa x0_aux khỏi danh sách biến phi cơ sở và danh sách tất cả biến
            self.non_basic_vars.remove(self.auxiliary_var_name)
            if self.auxiliary_var_name in self.all_vars_ordered:
                self.all_vars_ordered.remove(self.auxiliary_var_name)
        elif self.auxiliary_var_name in self.basic_vars:
            # Điều này không nên xảy ra nếu các bước đẩy x0_aux ra khỏi cơ sở ở trên thành công
            # Hoặc nếu nó là dòng dư thừa 0=0 đã bị loại bỏ.
            self._log(f"CRITICAL WARNING: Auxiliary variable '{self.auxiliary_var_name}' is still basic when preparing for Phase 2. This should have been handled.")
            # Nếu nó vẫn là cơ sở và giá trị của nó là 0 và dòng của nó là 0=0 (sau khi bỏ các biến khác), thì có thể chấp nhận được
            # và dòng đó có thể bị xóa. Tuy nhiên, logic trên đã cố gắng xử lý.
            # Coi như lỗi nếu nó vẫn là cơ sở ở đây.
            return self._extract_solution("ErrorPhase2_AuxInBasis"), self.logs


        # 3. Khôi phục hàm mục tiêu gốc (đã được chuẩn hóa thành min bởi wrapper)
        self.current_objective_key = 'z_original' # Key này đã được đặt trong super().__init__
        self.current_objective_type = "minimize" # Luôn là min sau chuẩn hóa
        
        # Biểu thức hàm mục tiêu gốc: Z = sum(original_coeffs * decision_vars)
        # self.original_problem_objective_coeffs là hệ số của hàm min (đã có thể bị nhân -1)
        z_original_expr_substituted: Dict[str, float] = {'const': 0.0}
        for i, decision_var_name in enumerate(self.decision_vars_names):
            coeff_orig = self.original_problem_objective_coeffs[i] if i < len(self.original_problem_objective_coeffs) else 0.0
            
            if decision_var_name in self.basic_vars: # Nếu biến quyết định là cơ sở
                # Thay thế nó bằng biểu thức từ từ điển
                basic_var_expr = self.dictionary.get(decision_var_name, {})
                z_original_expr_substituted['const'] += coeff_orig * basic_var_expr.get('const', 0.0)
                for nb_var, nb_coeff in basic_var_expr.items():
                    if nb_var != 'const': # Chỉ các biến phi cơ sở trong biểu thức của biến cơ sở
                        z_original_expr_substituted[nb_var] = z_original_expr_substituted.get(nb_var, 0.0) + coeff_orig * nb_coeff
            elif decision_var_name in self.non_basic_vars: # Nếu biến quyết định là phi cơ sở
                z_original_expr_substituted[decision_var_name] = z_original_expr_substituted.get(decision_var_name, 0.0) + coeff_orig
            # else: biến quyết định không có trong cơ sở cũng không phi cơ sở (lỗi)

        self.dictionary[self.current_objective_key] = z_original_expr_substituted
        self._log(f"Restored original objective '{self.current_objective_key}' for Phase 2.")
        self._log_dictionary(phase_info="Phase 2 - Initial")

        # --- PHA 2: Tối ưu hóa hàm mục tiêu gốc ---
        self.current_phase = 2
        self._log("--- Starting Phase 2 Simplex Iterations (Original Problem) ---")
        
        remaining_iterations_for_phase2 = max_iterations - self.iteration_count
        if remaining_iterations_for_phase2 <= 0:
             remaining_iterations_for_phase2 = max_iterations // 2 +1 if max_iterations > 0 else 1
        
        phase2_final_status = self._run_simplex_phase_loop(max_phase_iterations=remaining_iterations_for_phase2)
        self._log(f"Phase 2 (Original Problem) ended with status: {phase2_final_status}")

        # Trích xuất nghiệm cuối cùng dựa trên trạng thái của Pha 2
        final_solution = self._extract_solution(phase2_final_status)
        
        # Lưu ý: việc điều chỉnh giá trị hàm mục tiêu nếu bài toán gốc là MAX
        # sẽ được thực hiện bởi hàm bao bọc bên ngoài (solve_with_auxiliary_problem_simplex)
        return final_solution, self.logs


def solve_with_auxiliary_problem_simplex(
    problem_data_input: Dict[str, Any], # Đây là "Định dạng A"
    max_iterations_total=50
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Hàm bao bọc để giải bài toán LP bằng AuxiliaryProblemSolver.
    Sẽ chuẩn hóa bài toán trước khi giải.
    """
    overall_logs: List[str] = []
    overall_logs.append("--- solve_with_auxiliary_problem_simplex called ---")

    # Bước 1: Chuẩn hóa bài toán (mục tiêu min, ràng buộc <=)
    standardized_problem_data, was_maximized = standardize_problem_for_simplex(problem_data_input, overall_logs)

    if standardized_problem_data is None:
        overall_logs.append("ERROR: Standardization failed for AuxiliaryProblemSolver.")
        return {"status": "Error", "message": "Input data standardization failed."}, overall_logs
    
    # Lưu lại loại mục tiêu gốc để log và điều chỉnh kết quả sau này
    standardized_problem_data['objective_type_before_standardization'] = problem_data_input.get("objective", "N/A")


    # Bước 2: Khởi tạo và chạy solver với dữ liệu đã chuẩn hóa
    solver = AuxiliaryProblemSolver(standardized_problem_data)
    solution, solver_logs = solver.solve(max_iterations=max_iterations_total)
    overall_logs.extend(solver_logs)

    # Bước 3: Điều chỉnh lại giá trị hàm mục tiêu nếu bài toán gốc là Maximize
    if solution and solution.get("status") == "Optimal" and was_maximized:
        if solution.get("objective_value") is not None:
            solution["objective_value"] *= -1
            overall_logs.append(f"Final objective value (for original MAX problem, Auxiliary) adjusted: {solution['objective_value']:.4g}")
        else:
            overall_logs.append(f"Warning (Auxiliary): Solution status is Optimal but objective_value is None. Cannot adjust for original MAX problem.")

    elif solution and solution.get("status") == "Unbounded" and was_maximized:
        # Nếu bài toán min -Z không bị chặn dưới, bài toán max Z gốc cũng không bị chặn trên.
        overall_logs.append("Original MAX problem is also Unbounded (Auxiliary).")
    
    overall_logs.append("--- solve_with_auxiliary_problem_simplex finished ---")
    return solution, overall_logs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Bài toán từ hình ảnh image_f72cfa.jpg (Bài 1.20 trong hình đó)
    # GỐC: min x1 + x2
    # s.t.
    # -x1 + x2 <= -2  (1)
    #  x1 + 2x2 <= 4  (2)
    #  x1       <= 1  (3)
    # x1, x2 >= 0 (Các ràng buộc phi âm này cần được thêm vào input nếu muốn solver xử lý tường minh)
    # standardize_problem_for_simplex sẽ chuyển đổi x1>=0 thành -x1<=0, v.v.

    problem_f72cfa_input_A_format = {
        "objective": "min",
        "coeffs": [1, 1],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "R1_orig", "lhs": [-1, 1], "op": "<=", "rhs": -2}, # RHS âm
            {"name": "R2_orig", "lhs": [1, 2], "op": "<=", "rhs": 4},
            {"name": "R3_orig", "lhs": [1, 0], "op": "<=", "rhs": 1},
            {"name": "nonneg_x1", "lhs": [1,0], "op": ">=", "rhs": 0}, # Sẽ thành -x1 <= 0
            {"name": "nonneg_x2", "lhs": [0,1], "op": ">=", "rhs": 0}  # Sẽ thành -x2 <= 0
        ]
    }
    print("--- Solving Problem from image_f72cfa.jpg with AuxiliaryProblemSolver (Standardized Input) ---")
    solution_f72, logs_f72 = solve_with_auxiliary_problem_simplex(problem_f72cfa_input_A_format, max_iterations_total=20)

    print("\n--- FULL LOGS (image_f72cfa.jpg - Auxiliary) ---")
    for log_entry in logs_f72: print(log_entry)
    print("\n--- FINAL SOLUTION (image_f72cfa.jpg - Auxiliary) ---")
    if solution_f72: import json; print(json.dumps(solution_f72, indent=2))
    # Kết quả mong đợi từ hình ảnh là "Infeasible" (vì min x0 = 1/2 > 0).

    # Test một bài toán MAX có ràng buộc >=
    # Max Z = 2x1 + x2
    # s.t. x1 + x2 <= 10
    #      x1 >= 2      => -x1 <= -2
    #      x2 >= 1      => -x2 <= -1
    # x1,x2 >=0 (sẽ được thêm vào và chuẩn hóa)
    # Lời giải: x1=9, x2=1, Z_max = 19
    problem_max_geq_A_format = {
        "objective": "max",
        "coeffs": [2, 1],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "C1", "lhs": [1,1], "op": "<=", "rhs": 10},
            {"name": "C2", "lhs": [1,0], "op": ">=", "rhs": 2},
            {"name": "C3", "lhs": [0,1], "op": ">=", "rhs": 1},
            {"name": "nonneg_x1", "lhs": [1,0], "op": ">=", "rhs": 0},
            {"name": "nonneg_x2", "lhs": [0,1], "op": ">=", "rhs": 0},
        ]
    }
    print("\n\n--- Solving MAX problem with GEQ constraints (Auxiliary) ---")
    solution_max_geq, logs_max_geq = solve_with_auxiliary_problem_simplex(problem_max_geq_A_format, max_iterations_total=20)
    print("\n--- FULL LOGS (MAX GEQ - Auxiliary) ---")
    for log_entry in logs_max_geq: print(log_entry)
    print("\n--- FINAL SOLUTION (MAX GEQ - Auxiliary) ---")
    if solution_max_geq: import json; print(json.dumps(solution_max_geq, indent=2))


    # Test bài toán có ràng buộc "=="
    # Min 2x1 + 3x2
    # x1 + x2 == 5   => x1+x2 <= 5  VÀ  -x1-x2 <= -5
    # x1 >= 1        => -x1 <= -1
    # x2 <= 3
    problem_eq_A_format = {
        "objective": "min",
        "coeffs": [2, 3],
        "variables_names_for_title_only": ["x1", "x2"],
        "constraints": [
            {"name": "C1_eq", "lhs": [1,1], "op": "==", "rhs": 5},
            {"name": "C2_geq", "lhs": [1,0], "op": ">=", "rhs": 1},
            {"name": "C3_leq", "lhs": [0,1], "op": "<=", "rhs": 3},
            {"name": "nonneg_x1", "lhs": [1,0], "op": ">=", "rhs": 0}, # x1>=0 (dư thừa nếu có x1>=1)
            {"name": "nonneg_x2", "lhs": [0,1], "op": ">=", "rhs": 0}, # x2>=0
        ]
    }
    print("\n\n--- Solving MIN problem with EQ constraint (Auxiliary) ---")
    # Lời giải x1=2, x2=3, Z = 4+9 = 13
    solution_eq, logs_eq = solve_with_auxiliary_problem_simplex(problem_eq_A_format, max_iterations_total=20)
    print("\n--- FULL LOGS (MIN EQ - Auxiliary) ---")
    for log_entry in logs_eq: print(log_entry)
    print("\n--- FINAL SOLUTION (MIN EQ - Auxiliary) ---")
    if solution_eq: import json; print(json.dumps(solution_eq, indent=2))

