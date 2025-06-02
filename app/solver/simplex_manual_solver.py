# /app/solver/simplex_manual_solver_dict_format.py
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class SimplexDictionarySolver:
    """
    Lớp triển khai thuật toán Simplex sử dụng phương pháp từ điển (Dictionary Method).
    Mục tiêu là cung cấp log từng bước theo định dạng phương trình, giống như trong hình ảnh.
    """
    def __init__(self, problem_data: Dict[str, Any]):
        self.problem_data = problem_data
        self.logs = []
        self.objective_type = problem_data.get("objective", {}).get("type", "maximize").lower()
        self.iteration_count = 0
        
        # 'dictionary' sẽ lưu hệ phương trình
        # key: tên biến cơ sở (ví dụ: 'z', 'w1', 'w2')
        # value: một dict khác biểu diễn vế phải, ví dụ: {'const': 4, 'x1': -1, 'x2': -1}
        self.dictionary: Dict[str, Dict[str, float]] = {}
        
        self.decision_vars = problem_data.get("variables", [])
        self.slack_vars = []
        self.basic_vars: List[str] = [] # Các biến cơ sở
        self.non_basic_vars: List[str] = [] # Các biến phi cơ sở

    def _log(self, message: str, print_to_console: bool = True):
        """Ghi log và có thể in ra console."""
        self.logs.append(message)
        if print_to_console:
            print(message)

    def _format_expr(self, expr_dict: Dict[str, float]) -> str:
        """Định dạng một biểu thức từ điển thành chuỗi, ví dụ: 4 - x1 - x2"""
        parts = []
        const = expr_dict.get('const', 0)
        if const != 0 or not any(expr_dict.values()):
            parts.append(f"{const:.4g}") # .4g để định dạng số đẹp hơn

        for var, coeff in expr_dict.items():
            if var == 'const' or coeff == 0:
                continue
            
            # Định dạng hệ số và dấu
            abs_coeff = abs(coeff)
            sign = "-" if coeff < 0 else "+"
            
            if abs_coeff == 1:
                parts.append(f" {sign} {var}")
            else:
                parts.append(f" {sign} {abs_coeff:.4g}{var}")

        # Xử lý dấu '+' ở đầu nếu có
        result = " ".join(parts).strip()
        if result.startswith('+'):
            result = result[1:].strip()
        return result if result else "0"

    def _log_dictionary(self):
        """In ra từ điển hiện tại theo định dạng phương trình."""
        log_str = "Current Dictionary:\n"
        # In hàm mục tiêu trước
        if 'z' in self.dictionary:
            log_str += f"z' = {self._format_expr(self.dictionary['z'])}\n"
        
        # In các biến cơ sở
        for var in self.basic_vars:
            log_str += f"{var} = {self._format_expr(self.dictionary[var])}\n"
        
        self._log(log_str, print_to_console=True)

    def _build_initial_dictionary(self) -> bool:
        """Xây dựng từ điển ban đầu từ problem_data."""
        self._log("--- Building Initial Dictionary ---")

        # 1. Khởi tạo biến
        self.non_basic_vars = list(self.decision_vars)
        constraints = self.problem_data["constraints"]

        # 2. Xây dựng phương trình mục tiêu 'z'
        # Đối với bài toán Maximize z = c1x1 + c2x2, từ điển sẽ là z = 0 + c1x1 + c2x2
        obj_coeffs = self.problem_data["objective"]["coefficients"]
        z_expr = {'const': 0.0}
        for i, var in enumerate(self.decision_vars):
            z_expr[var] = obj_coeffs[i]
        self.dictionary['z'] = z_expr

        # 3. Xây dựng phương trình cho các ràng buộc
        for i, constr in enumerate(constraints):
            # Giả định ràng buộc là <= và vế phải không âm
            if constr["type"] != "<=" or constr["rhs"] < 0:
                self._log(f"Error: This solver only supports '<=' constraints with non-negative RHS.")
                return False
            
            slack_var = f"w{i+1}"
            self.slack_vars.append(slack_var)
            self.basic_vars.append(slack_var)

            # w_i = rhs - (a_i1*x1 + a_i2*x2 + ...)
            constr_expr = {'const': constr["rhs"]}
            for j, coeff in enumerate(constr["coefficients"]):
                constr_expr[self.decision_vars[j]] = -coeff
            self.dictionary[slack_var] = constr_expr
        
        self._log(f"Initial Basic Variables: {self.basic_vars}")
        self._log(f"Initial Non-Basic Variables: {self.non_basic_vars}")
        self._log_dictionary()
        return True

    def _select_entering_variable(self) -> Optional[str]:
        """Chọn biến vào cơ sở (có hệ số dương lớn nhất trong hàm mục tiêu)."""
        z_expr = self.dictionary['z']
        
        best_var = None
        max_coeff = 1e-9 # Một số dương rất nhỏ để tránh sai số
        
        for var, coeff in z_expr.items():
            if var != 'const' and coeff > max_coeff:
                max_coeff = coeff
                best_var = var
                
        if best_var:
            self._log(f"-> Entering variable (most positive in z'): {best_var} (coeff: {max_coeff:.4g})")
        else:
            self._log("Optimality condition met. No positive coefficients in z'.")
            
        return best_var

    def _select_leaving_variable(self, entering_var: str) -> Optional[str]:
        """Chọn biến ra khỏi cơ sở bằng kiểm tra tỷ lệ."""
        min_ratio = float('inf')
        leaving_var = None
        
        self._log(f"Calculating ratios for entering variable '{entering_var}':")
        
        for basic_var in self.basic_vars:
            expr = self.dictionary[basic_var]
            entering_coeff = expr.get(entering_var, 0)
            
            # Tỷ lệ chỉ được tính nếu hệ số của biến vào là âm
            if entering_coeff < -1e-9:
                ratio = expr['const'] / -entering_coeff
                self._log(f"  - Row '{basic_var}': ratio = {expr['const']:.4g} / {-entering_coeff:.4g} = {ratio:.4g}")
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_var = basic_var
        
        if leaving_var:
            self._log(f"-> Leaving variable (smallest non-negative ratio): {leaving_var} (ratio: {min_ratio:.4g})")
        else:
            self._log("Error: Problem is unbounded. No leaving variable found.")

        return leaving_var
        
    def _perform_pivot(self, entering_var: str, leaving_var: str):
        """Thực hiện phép xoay để cập nhật từ điển."""
        self._log(f"\n--- Pivoting: {entering_var} enters, {leaving_var} leaves ---\n")
        
        # 1. Lấy phương trình của biến rời khỏi cơ sở
        leaving_expr = self.dictionary.pop(leaving_var)
        leaving_coeff = leaving_expr[entering_var]
        
        # 2. Tạo phương trình mới cho biến vào cơ sở (giải theo entering_var)
        # entering_var = (const - leaving_var - ...) / -leaving_coeff
        entering_expr = {}
        for var, coeff in leaving_expr.items():
            if var != entering_var:
                entering_expr[var] = -coeff / leaving_coeff
        # Thêm biến rời đi vào vế phải
        entering_expr[leaving_var] = 1.0 / -leaving_coeff
        
        # 3. Thay thế entering_var trong tất cả các phương trình còn lại
        for basic_var, expr in self.dictionary.items():
            if entering_var in expr:
                factor = expr.pop(entering_var)
                # Thêm biểu thức của entering_var vào
                for var, coeff in entering_expr.items():
                    expr[var] = expr.get(var, 0) + factor * coeff
        
        # 4. Thêm phương trình mới vào từ điển
        self.dictionary[entering_var] = entering_expr
        
        # 5. Cập nhật danh sách biến cơ sở và phi cơ sở
        self.basic_vars.remove(leaving_var)
        self.basic_vars.append(entering_var)
        self.non_basic_vars.remove(entering_var)
        self.non_basic_vars.append(leaving_var)
        
        self._log_dictionary()

    def solve(self, max_iterations=20) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        if not self._build_initial_dictionary():
            return {"status": "Error", "message": "Failed to build initial dictionary."}, self.logs

        while self.iteration_count < max_iterations:
            self.iteration_count += 1
            self._log(f"\n>>>>>>>>> ITERATION {self.iteration_count} <<<<<<<<<<", print_to_console=True)
            
            entering_var = self._select_entering_variable()
            if not entering_var:
                # Đã đạt tối ưu
                return self._extract_solution(), self.logs
            
            leaving_var = self._select_leaving_variable(entering_var)
            if not leaving_var:
                # Bài toán không bị chặn
                return {"status": "Unbounded"}, self.logs
                
            self._perform_pivot(entering_var, leaving_var)
            
        return {"status": "MaxIterationsReached"}, self.logs

    def _extract_solution(self) -> Dict[str, Any]:
        """Trích xuất lời giải từ từ điển cuối cùng."""
        self._log("\n--- Final Solution ---")
        solution = {
            "status": "Optimal",
            "variables": {},
            "objective_value": self.dictionary['z']['const']
        }
        
        # Các biến quyết định
        for var in self.decision_vars:
            if var in self.basic_vars: # Nếu là biến cơ sở
                solution["variables"][var] = self.dictionary[var]['const']
            else: # Nếu là biến phi cơ sở
                solution["variables"][var] = 0.0
        
        self._log(f"Objective Value z = {solution['objective_value']:.4g}")
        for var, val in solution["variables"].items():
            self._log(f"{var} = {val:.4g}")

        # Chọn các biến phi cơ sở và gán bằng 0
        final_non_basic = " , ".join([f"{v} = 0" for v in self.non_basic_vars])
        self._log(f"Final non-basic variables: {final_non_basic}")
        
        return solution

def solve_with_simplex_manual(problem_data: Dict[str, Any], **kwargs) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Hàm bao bọc để gọi SimplexDictionarySolver."""
    if problem_data.get("objective", {}).get("type") == "minimize":
        # Chuyển bài toán min thành max(-z)
        problem_data["objective"]["coefficients"] = [-c for c in problem_data["objective"]["coefficients"]]
        problem_data["objective"]["type"] = "maximize"
        
    solver = SimplexDictionarySolver(problem_data)
    solution, logs = solver.solve(**kwargs)

    # Chuyển đổi lại giá trị hàm mục tiêu nếu là bài toán min
    if solver.problem_data.get("original_type") == "minimize" and solution and "objective_value" in solution:
        solution["objective_value"] *= -1

    return solution, logs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Bài toán từ hình ảnh của bạn
    # max z = 3x1 + 2x2 + 4x3
    # x1 + x2 + 2x3 <= 4  (w1)
    # 2x1 + x2 + 3x3 <= 5  (w2) -> Sử dụng phương trình này thay vì phương trình bị lỗi trong hình
    # 2x1 + x2 + 3x3 <= 7  (w3)
    problem_from_image = {
        "objective": {"type": "maximize", "coefficients": [3, 2, 4]},
        "variables": ["x1", "x2", "x3"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 1, 2], "type": "<=", "rhs": 4},
            {"name": "c2", "coefficients": [2, 1, 3], "type": "<=", "rhs": 5},
            {"name": "c3", "coefficients": [2, 1, 3], "type": "<=", "rhs": 7}, # Có vẻ ràng buộc c2 và c3 giống nhau
        ]
    }

    # Chạy bộ giải
    final_solution, all_logs = solve_with_simplex_manual(problem_from_image)

    # In kết quả cuối cùng một cách gọn gàng
    print("\n===================================")
    print("      FINAL SUMMARY")
    print("===================================")
    if final_solution:
        print(f"Status: {final_solution.get('status')}")
        if final_solution.get('status') == 'Optimal':
            print(f"Optimal Objective Value: {final_solution.get('objective_value'):.4g}")
            print("Variable Values:")
            for var, val in final_solution.get('variables', {}).items():
                print(f"  {var} = {val:.4g}")
    else:
        print("No solution found.")

