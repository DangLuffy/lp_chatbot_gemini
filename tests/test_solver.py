# /tests/test_solver.py

import pytest
from app.solver.pulp_cbc_solver import solve_with_pulp_cbc
from app.solver.simplex_manual_solver_dict_format import solve_with_simplex_manual

# Dữ liệu cho một bài toán tối đa hóa đơn giản
# Maximize 3x1 + 5x2
# s.t. x1 <= 4
#      2x2 <= 12
#      3x1 + 2x2 <= 18
# x1, x2 >= 0
# Lời giải: x1=2, x2=6, z=36
MAX_PROBLEM = {
    "objective": {"type": "maximize", "coefficients": [3, 5]},
    "variables": ["x1", "x2"],
    "constraints": [
        {"name": "c1", "coefficients": [1, 0], "type": "<=", "rhs": 4},
        {"name": "c2", "coefficients": [0, 2], "type": "<=", "rhs": 12},
        {"name": "c3", "coefficients": [3, 2], "type": "<=", "rhs": 18},
    ]
}
MAX_SOLUTION = {
    "status": "Optimal",
    "objective_value": 36.0,
    "variables": {"x1": 2.0, "x2": 6.0}
}

# Dữ liệu cho bài toán không khả thi (Infeasible)
# Maximize x1
# s.t. x1 >= 2
#      x1 <= 1
INFEASIBLE_PROBLEM = {
    "objective": {"type": "maximize", "coefficients": [1]},
    "variables": ["x1"],
    "constraints": [
        {"name": "c1", "coefficients": [1], "type": ">=", "rhs": 2},
        {"name": "c2", "coefficients": [1], "type": "<=", "rhs": 1},
    ]
}

# Dữ liệu cho bài toán không bị chặn (Unbounded)
# Maximize x1 + x2
# s.t. x1 - x2 >= 1
UNBOUNDED_PROBLEM = {
    "objective": {"type": "maximize", "coefficients": [1, 1]},
    "variables": ["x1", "x2"],
    "constraints": [
        {"name": "c1", "coefficients": [1, -1], "type": ">=", "rhs": 1},
    ]
}


def test_pulp_cbc_solver_maximize():
    """Kiểm tra bộ giải PuLP với bài toán maximize."""
    solution, _ = solve_with_pulp_cbc(MAX_PROBLEM)
    assert solution["status"] == MAX_SOLUTION["status"]
    assert solution["objective_value"] == pytest.approx(MAX_SOLUTION["objective_value"])
    for var, val in MAX_SOLUTION["variables"].items():
        assert solution["variables"][var] == pytest.approx(val)

def test_pulp_cbc_solver_infeasible():
    """Kiểm tra bộ giải PuLP với bài toán không khả thi."""
    solution, _ = solve_with_pulp_cbc(INFEASIBLE_PROBLEM)
    assert solution["status"] == "Infeasible"
    
def test_pulp_cbc_solver_unbounded():
    """Kiểm tra bộ giải PuLP với bài toán không bị chặn."""
    # Lưu ý: PuLP có thể không luôn phát hiện unbounded với các ràng buộc đơn giản
    # và có thể trả về lỗi hoặc trạng thái khác tùy thuộc vào bộ giải con.
    # Test này kiểm tra trạng thái Unbounded nếu được phát hiện.
    solution, _ = solve_with_pulp_cbc(UNBOUNDED_PROBLEM)
    assert solution["status"] == "Unbounded"

def test_simplex_manual_solver_maximize():
    """Kiểm tra bộ giải Simplex thủ công với bài toán maximize."""
    # Simplex thủ công yêu cầu tất cả ràng buộc là <=
    # Ta sẽ dùng bài toán max_problem vì nó đã ở dạng chuẩn
    solution, _ = solve_with_simplex_manual(MAX_PROBLEM.copy())
    assert solution["status"] == MAX_SOLUTION["status"]
    assert solution["objective_value"] == pytest.approx(MAX_SOLUTION["objective_value"])
    for var, val in MAX_SOLUTION["variables"].items():
        assert solution["variables"][var] == pytest.approx(val)

