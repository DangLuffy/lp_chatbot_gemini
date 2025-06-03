# /app/solver/base_simplex_dictionary_solver.py
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod # Vẫn giữ ABC cho solve và _build_initial_dictionary

logger = logging.getLogger(__name__)

class BaseSimplexDictionarySolver(ABC):
    """
    Lớp cơ sở cho các bộ giải Simplex sử dụng phương pháp từ điển.
    Cung cấp logic xoay chung và triển khai quy tắc Bland mặc định cho việc chọn biến.
    """
    def __init__(self, problem_data: Dict[str, Any], objective_key: str = 'z'):
        self.problem_data: Dict[str, Any] = problem_data
        self.logs: List[str] = []
        # objective_type được lấy từ problem_data, nhưng current_objective_type có thể thay đổi (ví dụ trong TwoPhase)
        self.original_objective_type: str = problem_data.get("objective", {}).get("type", "maximize").lower()
        self.current_objective_type: str = self.original_objective_type # Ban đầu giống nhau
        
        self.iteration_count: int = 0
        self.dictionary: Dict[str, Dict[str, float]] = {} 
        
        self.decision_vars_names: List[str] = list(problem_data.get("variables", []))
        self.slack_vars_names: List[str] = [] 
        self.surplus_vars_names: List[str] = [] 
        self.artificial_vars_names: List[str] = [] 
        
        self.all_vars_ordered: List[str] = [] 
        self.basic_vars: List[str] = [] 
        self.non_basic_vars: List[str] = [] 
        self.epsilon: float = 1e-9 
        self.current_objective_key: str = objective_key

        self._build_initial_dictionary_common_setup()

    def _log(self, message: str, print_to_console: bool = False):
        self.logs.append(message)
        if print_to_console:
            logger.info(message)

    def _format_expr(self, expr_dict: Dict[str, float]) -> str:
        # ... (Giữ nguyên như phiên bản trước) ...
        parts = []
        const_val = expr_dict.get('const', 0.0)
        if abs(const_val) > self.epsilon or not any(abs(val) > self.epsilon for name, val in expr_dict.items() if name != 'const'):
            parts.append(f"{const_val:.4g}")
        
        sorted_vars = sorted(
            [var for var in expr_dict if var != 'const' and abs(expr_dict[var]) > self.epsilon],
            key=lambda v_name: self.all_vars_ordered.index(v_name) if v_name in self.all_vars_ordered else float('inf')
        )
        for var_name in sorted_vars:
            coeff = expr_dict[var_name]
            abs_coeff = abs(coeff); current_sign = "-" if coeff < 0 else "+"
            term_str = f"{abs_coeff:.4g}{var_name}" if abs(abs_coeff - 1.0) >= self.epsilon else var_name
            if not parts or (len(parts) == 1 and (parts[0] == "0" or parts[0] == "0.0")):
                parts = [f"{current_sign.strip()}{term_str}"] if current_sign == "-" else [term_str]
            else: parts.append(f" {current_sign} {term_str}")
        if not parts: return "0"
        result = " ".join(parts).strip();
        if result.startswith("+ ") and len(result) > 2 and result[2].isalpha(): result = result[2:]
        return result

    def _log_dictionary(self, phase_info: Optional[str] = None):
        # ... (Giữ nguyên như phiên bản trước) ...
        phase_str = f" ({phase_info})" if phase_info else ""
        obj_key_display = self.current_objective_key
        log_str = f"--- Iteration {self.iteration_count}{phase_str} ---\nCurrent Dictionary (Objective: {self.current_objective_type} {obj_key_display}):\n"
        if obj_key_display in self.dictionary:
            log_str += f"{obj_key_display} = {self._format_expr(self.dictionary[obj_key_display])}\n"
        sorted_basic_vars = sorted(self.basic_vars, key=lambda v: self.all_vars_ordered.index(v) if v in self.all_vars_ordered else float('inf'))
        for var_name in sorted_basic_vars:
            if var_name in self.dictionary:
                 log_str += f"{var_name} = {self._format_expr(self.dictionary[var_name])}\n"
            else: self._log(f"Warning: Basic variable '{var_name}' not found in dictionary during logging.")
        self._log(log_str)


    def _build_initial_dictionary_common_setup(self):
        self.all_vars_ordered.extend(self.decision_vars_names)
        self.non_basic_vars = list(self.decision_vars_names)

    @abstractmethod
    def _build_initial_dictionary(self) -> bool:
        """Phải được triển khai bởi lớp con."""
        pass

    def _select_entering_variable(self) -> Optional[str]:
        """Chọn biến vào cơ sở theo Quy tắc Bland (mặc định)."""
        obj_expr = self.dictionary.get(self.current_objective_key)
        if obj_expr is None:
            self._log(f"ERROR: Objective key '{self.current_objective_key}' not found in dictionary.")
            return None
            
        candidate_entering_vars: List[str] = []

        if self.current_objective_type == "maximize":
            for var_name in self.non_basic_vars: # Chỉ xét các biến phi cơ sở
                if obj_expr.get(var_name, 0.0) > self.epsilon:
                    candidate_entering_vars.append(var_name)
        else: # current_objective_type == "minimize"
            for var_name in self.non_basic_vars: # Chỉ xét các biến phi cơ sở
                if obj_expr.get(var_name, 0.0) < -self.epsilon:
                    candidate_entering_vars.append(var_name)
        
        if not candidate_entering_vars:
            self._log(f"Optimality condition met for objective '{self.current_objective_key}'. No more candidates for entering variable.")
            return None # Đã tối ưu hoặc không có ứng viên
        
        # Quy tắc Bland: Chọn biến có chỉ số nhỏ nhất
        candidate_entering_vars.sort(key=lambda v_name: self.all_vars_ordered.index(v_name))
        entering_var = candidate_entering_vars[0]
        
        coeff_val_in_obj = obj_expr.get(entering_var, 0.0)
        self._log(f"Selected Entering (Bland): {entering_var} (coeff in obj '{self.current_objective_key}': {coeff_val_in_obj:.4g}, index: {self.all_vars_ordered.index(entering_var)})")
        return entering_var

    def _select_leaving_variable(self, entering_var: str) -> Optional[str]:
        """Chọn biến ra khỏi cơ sở theo Quy tắc Bland (mặc định)."""
        min_ratio = float('inf')
        candidate_leaving_vars_for_min_ratio: List[str] = [] 
        
        self._log(f"Calculating ratios for entering variable '{entering_var}':")
        
        for basic_var_name in self.basic_vars:
            # Không cho biến mục tiêu hiện tại ra khỏi cơ sở (nếu nó lỡ vào)
            if basic_var_name == self.current_objective_key: 
                continue 
            
            basic_var_expr = self.dictionary.get(basic_var_name)
            if basic_var_expr is None:
                self._log(f"Warning: Basic variable '{basic_var_name}' not in dictionary during leaving variable selection.")
                continue

            coeff_entering_in_row = basic_var_expr.get(entering_var, 0.0) 
            
            if coeff_entering_in_row < -self.epsilon: # Cần hệ số âm của biến vào trong dòng ràng buộc
                constant_term = basic_var_expr.get('const', 0.0)
                # Tỷ lệ: hằng số / (-hệ số của biến vào)
                ratio = constant_term / (-coeff_entering_in_row)
                
                if ratio >= -self.epsilon: # Chỉ xét tỷ lệ không âm (cho phép bằng 0)
                    self._log(f"  - Row '{basic_var_name}': const={constant_term:.4g}, coeff_entering={coeff_entering_in_row:.4g}, ratio = {ratio:.4g}")
                    if ratio < min_ratio - self.epsilon: 
                        min_ratio = ratio
                        candidate_leaving_vars_for_min_ratio = [basic_var_name]
                    elif abs(ratio - min_ratio) < self.epsilon: 
                        candidate_leaving_vars_for_min_ratio.append(basic_var_name)
        
        if not candidate_leaving_vars_for_min_ratio: 
            self._log(f"Problem may be UNBOUNDED (no non-negative ratios for entering variable '{entering_var}').")
            return None 
        
        # Quy tắc Bland: Chọn biến có chỉ số nhỏ nhất
        candidate_leaving_vars_for_min_ratio.sort(key=lambda v_name: self.all_vars_ordered.index(v_name))
        leaving_var = candidate_leaving_vars_for_min_ratio[0]
        
        self._log(f"Selected Leaving (Bland): {leaving_var} (min_ratio: {min_ratio:.4g}, index: {self.all_vars_ordered.index(leaving_var)})")
        return leaving_var

    def _perform_pivot(self, entering_var: str, leaving_var: str) -> bool:
        # ... (Giữ nguyên như phiên bản trước) ...
        self._log(f"Pivoting: Variable '{entering_var}' enters basis, '{leaving_var}' leaves basis.")
        if leaving_var not in self.dictionary or entering_var not in self.dictionary[leaving_var]:
            self._log(f"ERROR: Pivot error. Leaving var '{leaving_var}' or entering var '{entering_var}' not correctly set up in dictionary."); return False
        leaving_var_expr = self.dictionary.pop(leaving_var); pivot_element_coeff = leaving_var_expr[entering_var] 
        if abs(pivot_element_coeff) < self.epsilon:
            self._log(f"ERROR: Pivot element for '{entering_var}' in '{leaving_var}' row is zero."); self.dictionary[leaving_var] = leaving_var_expr; return False
        new_entering_var_expr: Dict[str, float] = {}
        new_entering_var_expr['const'] = leaving_var_expr.get('const', 0.0) / (-pivot_element_coeff)
        new_entering_var_expr[leaving_var] = 1.0 / (-pivot_element_coeff)
        for var_name, coeff_val in leaving_var_expr.items():
            if var_name != 'const' and var_name != entering_var: new_entering_var_expr[var_name] = coeff_val / (-pivot_element_coeff)
        for other_basic_var_key in list(self.dictionary.keys()): 
            current_row_expr = self.dictionary[other_basic_var_key]
            coeff_of_entering_in_current_row = current_row_expr.pop(entering_var, 0.0) 
            if abs(coeff_of_entering_in_current_row) > self.epsilon: 
                current_row_expr['const'] = current_row_expr.get('const', 0.0) + coeff_of_entering_in_current_row * new_entering_var_expr['const']
                for var_in_new_expr, coeff_in_new_expr in new_entering_var_expr.items():
                    if var_in_new_expr != 'const':
                        current_row_expr[var_in_new_expr] = current_row_expr.get(var_in_new_expr, 0.0) + coeff_of_entering_in_current_row * coeff_in_new_expr
        self.dictionary[entering_var] = new_entering_var_expr
        self.basic_vars.remove(leaving_var); self.basic_vars.append(entering_var)
        self.non_basic_vars.remove(entering_var); self.non_basic_vars.append(leaving_var)
        return True

    def _extract_solution(self, final_status: str) -> Dict[str, Any]: # Thêm final_status
        self._log(f"Extracting solution from final dictionary. Status: {final_status}")
        solution: Dict[str, Any] = {
            "status": final_status, 
            "variables": {},
            # Chỉ lấy objective_value nếu tối ưu hoặc không bị chặn (và có giá trị)
            "objective_value": None 
        }
        if final_status == "Optimal":
            solution["objective_value"] = self.dictionary.get(self.current_objective_key, {}).get('const', 0.0)
        
        for var_name in self.decision_vars_names:
            if var_name in self.basic_vars: 
                solution["variables"][var_name] = self.dictionary.get(var_name, {}).get('const', 0.0)
            else: 
                solution["variables"][var_name] = 0.0
        
        if solution["objective_value"] is not None:
            self._log(f"Final Objective Value {self.current_objective_key} = {solution['objective_value']:.4g}")
        for var, val in solution["variables"].items(): self._log(f"Final {var} = {val:.4g}")
        return solution

    @abstractmethod
    def solve(self, max_iterations: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Phải được triển khai bởi lớp con."""
        pass

