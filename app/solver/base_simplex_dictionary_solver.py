# /app/solver/base_simplex_dictionary_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseSimplexDictionarySolver(ABC):
    """
    Lớp cơ sở cho các bộ giải Simplex sử dụng phương pháp từ điển.
    Cung cấp logic xoay chung và triển khai quy tắc Bland mặc định cho việc chọn biến.
    Lớp này mong đợi problem_data đầu vào đã được chuẩn hóa bởi hàm bao bọc:
    - "objective": "min"
    - "coeffs": list of objective coefficients for the "min" problem
    - "variables_names_for_title_only": ordered list of variable names
    - "constraints": list of constraints, where each constraint.op is "<="
                     (nếu solver này chỉ xử lý biến bù đơn giản).
    """
    def __init__(self, problem_data_standardized: Dict[str, Any], objective_key_in_dict: str = 'z_obj'):
        self.problem_data: Dict[str, Any] = problem_data_standardized # Dữ liệu đã được chuẩn hóa
        self.logs: List[str] = []

        # Dữ liệu đầu vào đã được chuẩn hóa, nên objective sẽ là "min"
        self.current_objective_type: str = self.problem_data.get("objective", "min").lower()
        if self.current_objective_type != "min":
            # This should ideally not happen if data is pre-standardized by the wrapper
            self._log(f"CRITICAL WARNING: BaseSimplexDictionarySolver received non-minimization problem type '{self.current_objective_type}' despite expecting standardized input. Proceeding as minimization.")
            self.current_objective_type = "min"

        # objective_coeffs_list là hệ số của hàm "min" (đã có thể bị nhân -1 nếu gốc là max)
        self.objective_coeffs_list: List[float] = self.problem_data.get("coeffs", [])
        self.decision_vars_names: List[str] = list(self.problem_data.get("variables_names_for_title_only", []))

        if not self.decision_vars_names and self.objective_coeffs_list:
            self.decision_vars_names = [f"x{i+1}" for i in range(len(self.objective_coeffs_list))]
            self._log(f"Warning: No variable names provided to BaseSimplex, auto-generating: {self.decision_vars_names}")
        elif len(self.decision_vars_names) != len(self.objective_coeffs_list):
            self._log(f"ERROR: Mismatch between variable names ({len(self.decision_vars_names)}) and objective coeffs ({len(self.objective_coeffs_list)}) in BaseSimplex.")
            # Consider raising an error here if strictness is required

        self.iteration_count: int = 0
        self.dictionary: Dict[str, Dict[str, float]] = {}

        self.slack_vars_names: List[str] = []
        self.surplus_vars_names: List[str] = [] # Thường không dùng nếu ràng buộc đã là <=
        self.artificial_vars_names: List[str] = [] # Thường được xử lý bởi solver chuyên biệt hơn

        self.all_vars_ordered: List[str] = []
        self.basic_vars: List[str] = []
        self.non_basic_vars: List[str] = []
        self.epsilon: float = 1e-9
        self.current_objective_key: str = objective_key_in_dict

        self._build_initial_dictionary_common_setup()

    def _log(self, message: str, print_to_console: bool = False):
        self.logs.append(message)
        if print_to_console: # For debugging during development
            logger.info(f"BaseSimplex: {message}")


    def _format_expr(self, expr_dict: Dict[str, float]) -> str:
        parts = []
        const_val = expr_dict.get('const', 0.0)
        # Hiển thị hằng số nếu nó khác 0 hoặc nếu không có biến nào khác trong biểu thức
        if abs(const_val) > self.epsilon or not any(abs(val) > self.epsilon for name, val in expr_dict.items() if name != 'const'):
            parts.append(f"{const_val:.4g}")

        # Sắp xếp các biến để hiển thị nhất quán
        sorted_vars = sorted(
            [var for var in expr_dict if var != 'const' and abs(expr_dict[var]) > self.epsilon],
            key=lambda v_name: self.all_vars_ordered.index(v_name) if v_name in self.all_vars_ordered else float('inf')
        )

        for var_name in sorted_vars:
            coeff = expr_dict[var_name]
            abs_coeff = abs(coeff)
            current_sign = "-" if coeff < 0 else "+"
            
            term_str = ""
            if abs(abs_coeff - 1.0) < self.epsilon: # Hệ số là 1 hoặc -1
                term_str = var_name
            else:
                term_str = f"{abs_coeff:.4g}{var_name}"

            if not parts or (len(parts) == 1 and (parts[0].strip() == "0" or parts[0].strip() == "0.0")): # Nếu phần tử đầu tiên là 0
                 parts = [f"{current_sign.strip()}{term_str}"] if current_sign == "-" else [term_str]
            else:
                parts.append(f" {current_sign} {term_str}")
        
        if not parts: return "0"
        result = "".join(parts).strip() # Dùng join rỗng rồi strip
        if result.startswith("+") and len(result) > 1: # Bỏ dấu + ở đầu nếu có
            result = result[1:].strip()
        return result


    def _log_dictionary(self, phase_info: Optional[str] = None):
        phase_str = f" ({phase_info})" if phase_info else ""
        obj_key_display = self.current_objective_key
        # self.current_objective_type đã được đặt là "min" trong __init__
        log_str = f"--- Iteration {self.iteration_count}{phase_str} ---\nCurrent Dictionary (Objective: {self.current_objective_type} {obj_key_display}):\n"
        
        if obj_key_display in self.dictionary:
            log_str += f"{obj_key_display} = {self._format_expr(self.dictionary[obj_key_display])}\n"
        else:
            log_str += f"{obj_key_display} = [Not yet fully defined in dictionary]\n"


        sorted_basic_vars_for_log = sorted(
            [bv for bv in self.basic_vars if bv in self.dictionary and bv != obj_key_display],
            key=lambda v: self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf')
        )

        for var_name in sorted_basic_vars_for_log:
            log_str += f"{var_name} = {self._format_expr(self.dictionary[var_name])}\n"
        
        # Log các biến cơ sở không tìm thấy trong dictionary (nếu có, để debug)
        for var_name in self.basic_vars:
            if var_name != obj_key_display and var_name not in self.dictionary :
                 self._log(f"Warning: Basic variable '{var_name}' is in self.basic_vars but not found in self.dictionary during logging.")

        self._log(log_str)


    def _build_initial_dictionary_common_setup(self):
        # self.decision_vars_names được lấy từ problem_data_standardized["variables_names_for_title_only"]
        self.all_vars_ordered.extend(self.decision_vars_names)
        self.non_basic_vars = list(self.decision_vars_names) # Ban đầu tất cả biến quyết định là phi cơ sở

    @abstractmethod
    def _build_initial_dictionary(self) -> bool:
        """
        Xây dựng từ điển ban đầu từ self.problem_data (đã được chuẩn hóa).
        Phải được triển khai bởi lớp con.
        Lớp con sẽ thêm biến bù, biến nhân tạo nếu cần.
        """
        pass

    def _select_entering_variable(self) -> Optional[str]:
        """
        Chọn biến vào cơ sở theo Quy tắc Bland (mặc định).
        Vì hàm mục tiêu đã được chuẩn hóa thành "minimize", tìm hệ số âm nhất.
        """
        obj_expr = self.dictionary.get(self.current_objective_key)
        if obj_expr is None:
            self._log(f"ERROR: Objective key '{self.current_objective_key}' not found in dictionary for selecting entering variable.")
            return None

        candidate_entering_vars: List[str] = []
        # self.current_objective_type sẽ luôn là "min"
        for var_name in self.non_basic_vars: # Chỉ xét các biến phi cơ sở
            if var_name not in self.all_vars_ordered: continue # Đảm bảo biến có trong danh sách
            coeff = obj_expr.get(var_name, 0.0)
            if coeff < -self.epsilon: # Cho hàm min, tìm hệ số âm
                candidate_entering_vars.append(var_name)

        if not candidate_entering_vars:
            self._log(f"Optimality condition met for objective '{self.current_objective_key}' (minimize). No more candidates for entering variable.")
            return None

        # Quy tắc Bland: Chọn biến có chỉ số nhỏ nhất (theo thứ tự trong all_vars_ordered)
        candidate_entering_vars.sort(key=lambda v_name: self.all_vars_ordered.index(v_name))
        entering_var = candidate_entering_vars[0]

        coeff_val_in_obj = obj_expr.get(entering_var, 0.0)
        self._log(f"Selected Entering (Bland for Min Objective): {entering_var} (coeff in obj '{self.current_objective_key}': {coeff_val_in_obj:.4g}, index: {self.all_vars_ordered.index(entering_var)})")
        return entering_var

    def _select_leaving_variable(self, entering_var: str) -> Optional[str]:
        """Chọn biến ra khỏi cơ sở theo Quy tắc Bland (mặc định)."""
        min_ratio = float('inf')
        candidate_leaving_vars_for_min_ratio: List[str] = []

        self._log(f"Calculating ratios for entering variable '{entering_var}':")

        # Sắp xếp các biến cơ sở để đảm bảo tính nhất quán khi chọn biến rời nếu có nhiều tỷ lệ bằng nhau (mặc dù Bland sẽ xử lý sau)
        sorted_basic_vars = sorted(
            [bv for bv in self.basic_vars if bv in self.all_vars_ordered],
            key=lambda v_name: self.all_vars_ordered.index(v_name)
            )

        for basic_var_name in sorted_basic_vars:
            if basic_var_name == self.current_objective_key: # Không cho hàm mục tiêu ra
                continue

            basic_var_expr = self.dictionary.get(basic_var_name)
            if basic_var_expr is None:
                self._log(f"Warning: Basic variable '{basic_var_name}' not in dictionary during leaving variable selection.")
                continue

            coeff_entering_in_row = basic_var_expr.get(entering_var, 0.0)

            # Chỉ xét khi hệ số của biến vào trong dòng ràng buộc là âm (để khi chia không đổi dấu bất đẳng thức)
            if coeff_entering_in_row < -self.epsilon:
                constant_term = basic_var_expr.get('const', 0.0)
                # Tỷ lệ: hằng số / (-hệ số của biến vào)
                ratio = constant_term / (-coeff_entering_in_row)

                # Chỉ xét tỷ lệ không âm (cho phép bằng 0 để xử lý suy biến)
                if ratio >= -self.epsilon: # Dùng -epsilon để bao gồm cả trường hợp bằng 0
                    self._log(f"  - Row '{basic_var_name}': const={constant_term:.4g}, coeff_entering={coeff_entering_in_row:.4g}, ratio = {ratio:.4g}")
                    if ratio < min_ratio - self.epsilon: # Tìm tỷ lệ nhỏ nhất
                        min_ratio = ratio
                        candidate_leaving_vars_for_min_ratio = [basic_var_name]
                    elif abs(ratio - min_ratio) < self.epsilon: # Nếu tỷ lệ bằng nhau, thêm vào danh sách ứng viên
                        candidate_leaving_vars_for_min_ratio.append(basic_var_name)
            # else: Nếu coeff_entering_in_row >= 0, biến vào không thể làm tăng giá trị biến cơ sở này (đối với min objective) hoặc làm nó âm.
            # Nếu coeff_entering_in_row > 0, tăng entering_var sẽ làm giảm basic_var_name. Không giới hạn.
            # Nếu coeff_entering_in_row = 0, entering_var không ảnh hưởng.

        if not candidate_leaving_vars_for_min_ratio:
            self._log(f"Problem may be UNBOUNDED (no non-negative ratios with negative coefficients for entering variable '{entering_var}').")
            return None

        # Quy tắc Bland: Nếu có nhiều biến cùng tỷ lệ nhỏ nhất, chọn biến có chỉ số nhỏ nhất
        candidate_leaving_vars_for_min_ratio.sort(key=lambda v_name: self.all_vars_ordered.index(v_name))
        leaving_var = candidate_leaving_vars_for_min_ratio[0]

        self._log(f"Selected Leaving (Bland): {leaving_var} (min_ratio: {min_ratio:.4g}, index: {self.all_vars_ordered.index(leaving_var)})")
        return leaving_var

    def _perform_pivot(self, entering_var: str, leaving_var: str) -> bool:
        self._log(f"Pivoting: Variable '{entering_var}' enters basis, '{leaving_var}' leaves basis.")
        
        # Kiểm tra leaving_var và entering_var có hợp lệ không
        if leaving_var not in self.dictionary:
            self._log(f"ERROR: Pivot error. Leaving var '{leaving_var}' not found in dictionary."); return False
        
        leaving_var_expr = self.dictionary[leaving_var] # Không pop() ở đây vội
        
        pivot_element_coeff = leaving_var_expr.get(entering_var)
        if pivot_element_coeff is None or abs(pivot_element_coeff) < self.epsilon:
            self._log(f"ERROR: Pivot element for '{entering_var}' in '{leaving_var}' row is zero or missing. Coeff: {pivot_element_coeff}");
            return False

        # 1. Xóa dòng của leaving_var khỏi từ điển (sẽ được thay thế bằng dòng của entering_var)
        self.dictionary.pop(leaving_var)

        # 2. Tạo dòng mới cho biến vào (entering_var)
        new_entering_var_expr: Dict[str, float] = {}
        # Hằng số: const_mới = const_cũ_dòng_rời / (-hệ_số_xoay)
        new_entering_var_expr['const'] = leaving_var_expr.get('const', 0.0) / (-pivot_element_coeff)
        # Biến rời trở thành phi cơ sở: coeff = 1 / (-hệ_số_xoay)
        new_entering_var_expr[leaving_var] = 1.0 / (-pivot_element_coeff)
        # Các biến phi cơ sở khác trong dòng rời: coeff_mới = coeff_cũ / (-hệ_số_xoay)
        for var_name, coeff_val in leaving_var_expr.items():
            if var_name != 'const' and var_name != entering_var: # Không thêm chính entering_var
                new_entering_var_expr[var_name] = coeff_val / (-pivot_element_coeff)

        # 3. Cập nhật các dòng còn lại trong từ điển (bao gồm cả dòng mục tiêu)
        for other_basic_var_key in list(self.dictionary.keys()): # list() để tránh lỗi thay đổi dict khi lặp
            current_row_expr = self.dictionary[other_basic_var_key]
            # Lấy hệ số của entering_var trong dòng hiện tại (nếu có) và xóa nó khỏi biểu thức
            coeff_of_entering_in_current_row = current_row_expr.pop(entering_var, 0.0)

            if abs(coeff_of_entering_in_current_row) > self.epsilon: # Nếu entering_var có mặt trong dòng này
                # Cập nhật hằng số: const_mới = const_cũ + coeff_entering_cũ_dòng_này * const_dòng_xoay_mới
                current_row_expr['const'] = current_row_expr.get('const', 0.0) + \
                                           coeff_of_entering_in_current_row * new_entering_var_expr['const']
                # Cập nhật các hệ số khác dựa trên dòng pivot mới (new_entering_var_expr)
                for var_in_new_expr, coeff_in_new_expr in new_entering_var_expr.items():
                    if var_in_new_expr != 'const': # Chỉ xử lý các biến, không phải hằng số
                        current_row_expr[var_in_new_expr] = current_row_expr.get(var_in_new_expr, 0.0) + \
                                                            coeff_of_entering_in_current_row * coeff_in_new_expr
        
        # 4. Thêm dòng mới của entering_var vào từ điển
        self.dictionary[entering_var] = new_entering_var_expr

        # 5. Cập nhật danh sách biến cơ sở và phi cơ sở
        if leaving_var in self.basic_vars: self.basic_vars.remove(leaving_var)
        if entering_var not in self.basic_vars: self.basic_vars.append(entering_var)
        
        if entering_var in self.non_basic_vars: self.non_basic_vars.remove(entering_var)
        if leaving_var not in self.non_basic_vars: self.non_basic_vars.append(leaving_var)
        
        # Sắp xếp lại basic_vars và non_basic_vars theo all_vars_ordered để log và xử lý nhất quán
        self.basic_vars.sort(key=lambda v: self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))
        self.non_basic_vars.sort(key=lambda v: self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))

        return True

    def _extract_solution(self, final_status: str) -> Dict[str, Any]:
        self._log(f"Extracting solution from final dictionary. Status: {final_status}")
        solution: Dict[str, Any] = {
            "status": final_status,
            "variables": {}, # Sẽ chỉ chứa các biến quyết định ban đầu
            "objective_value": None
        }

        # Lấy giá trị hàm mục tiêu từ từ điển nếu tối ưu
        if final_status == "Optimal" and self.current_objective_key in self.dictionary:
            solution["objective_value"] = self.dictionary[self.current_objective_key].get('const', 0.0)
        elif final_status == "Optimal" and self.current_objective_key not in self.dictionary:
             self._log(f"Warning: Status is Optimal but objective key '{self.current_objective_key}' not in dictionary. Objective value cannot be extracted from dictionary constant.")

        # Lấy giá trị cho các biến quyết định ban đầu
        for var_name in self.decision_vars_names:
            if var_name in self.basic_vars and var_name in self.dictionary:
                solution["variables"][var_name] = self.dictionary[var_name].get('const', 0.0)
            elif var_name in self.non_basic_vars: # Biến phi cơ sở có giá trị bằng 0
                solution["variables"][var_name] = 0.0
            else:
                # Biến quyết định không có trong cơ sở cũng không trong phi cơ sở
                # Điều này không nên xảy ra nếu all_vars_ordered được quản lý đúng
                # Có thể nó là biến không được sử dụng hoặc lỗi logic. Mặc định là 0.
                solution["variables"][var_name] = 0.0
                self._log(f"Warning: Decision variable '{var_name}' was not found in basic or non-basic lists during solution extraction. Defaulting its value to 0.")

        # Log thông tin
        if solution["objective_value"] is not None:
            self._log(f"Objective value from dictionary ({self.current_objective_key}) = {solution['objective_value']:.4g}")
        self._log("Final values for decision variables:")
        for var, val in solution["variables"].items(): self._log(f"  - {var} = {val:.4g}")
        
        return solution

    @abstractmethod
    def solve(self, max_iterations: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Phương thức chính để giải bài toán. Phải được triển khai bởi lớp con.
        Sẽ bao gồm logic cho các pha (nếu cần) và vòng lặp Simplex chính.
        """
        pass
