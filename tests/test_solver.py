# /tests/test_solver.py

import pytest
import os
import base64

from app.solver.pulp_cbc_solver import solve_with_pulp_cbc
from app.solver.simplex_manual_solver_dict_format import solve_with_simplex_manual
from app.solver.geometric_solver import solve_with_geometric_method

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
        {"name": "c1_x1_le_4", "coefficients": [1, 0], "type": "<=", "rhs": 4},
        {"name": "c2_x2_le_6", "coefficients": [0, 2], "type": "<=", "rhs": 12},
        {"name": "c3_3x1_2x2_le_18", "coefficients": [3, 2], "type": "<=", "rhs": 18},
    ]
}
MAX_SOLUTION = {
    "status": "Optimal",
    "objective_value": 36.0,
    "variables": {"x1": 2.0, "x2": 6.0}
}

# Dữ liệu cho bài toán không khả thi (Infeasible)
INFEASIBLE_PROBLEM = {
    "objective": {"type": "maximize", "coefficients": [1]},
    "variables": ["x1"],
    "constraints": [
        {"name": "c1", "coefficients": [1], "type": ">=", "rhs": 2},
        {"name": "c2", "coefficients": [1], "type": "<=", "rhs": 1},
    ]
}

# Dữ liệu cho bài toán không bị chặn (Unbounded)
UNBOUNDED_PROBLEM = {
    "objective": {"type": "maximize", "coefficients": [1, 1]},
    "variables": ["x1", "x2"],
    "constraints": [
        {"name": "c1", "coefficients": [1, -1], "type": ">=", "rhs": 1},
    ]
}


def test_pulp_cbc_solver_maximize():
    solution, _ = solve_with_pulp_cbc(MAX_PROBLEM)
    assert solution["status"] == MAX_SOLUTION["status"]
    assert solution["objective_value"] == pytest.approx(MAX_SOLUTION["objective_value"])
    for var, val in MAX_SOLUTION["variables"].items():
        assert solution["variables"][var] == pytest.approx(val)

def test_pulp_cbc_solver_infeasible():
    solution, _ = solve_with_pulp_cbc(INFEASIBLE_PROBLEM)
    assert solution["status"] == "Infeasible"
    
def test_pulp_cbc_solver_unbounded():
    solution, _ = solve_with_pulp_cbc(UNBOUNDED_PROBLEM)
    assert solution["status"] == "Unbounded"

def test_simplex_manual_solver_maximize():
    solution, _ = solve_with_simplex_manual(MAX_PROBLEM.copy())
    assert solution["status"] == MAX_SOLUTION["status"]
    assert solution["objective_value"] == pytest.approx(MAX_SOLUTION["objective_value"])
    for var, val in MAX_SOLUTION["variables"].items():
        assert solution["variables"][var] == pytest.approx(val)

def test_geometric_solver_maximize_and_save_plot():
    """Kiểm tra bộ giải hình học VÀ lưu ảnh kết quả để xem."""
    solution, _ = solve_with_geometric_method(MAX_PROBLEM)
    
    # Kiểm tra tính đúng đắn của kết quả
    assert solution["status"] == MAX_SOLUTION["status"]
    assert solution["objective_value"] == pytest.approx(MAX_SOLUTION["objective_value"])
    for var, val in MAX_SOLUTION["variables"].items():
        assert solution["variables"][var] == pytest.approx(val)

    # --- PHẦN THÊM MỚI: LƯU ẢNH RA FILE ---
    assert "plot_image_base64" in solution, "Missing plot image data in solution"
    
    # 1. Tạo thư mục output nếu chưa có
    output_dir = "test_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Lấy chuỗi base64 và tách phần header
    img_data_str = solution["plot_image_base64"]
    try:
        header, encoded = img_data_str.split(",", 1)
    except ValueError:
        pytest.fail("Base64 string format is incorrect.")
    
    # 3. Giải mã chuỗi base64 thành dữ liệu nhị phân
    image_data = base64.b64decode(encoded)
    
    # 4. Ghi dữ liệu nhị phân vào một file .png
    output_path = os.path.join(output_dir, "geometric_solver_test_result.png")
    with open(output_path, "wb") as f:
        f.write(image_data)
    
    # In ra đường dẫn để người dùng biết ảnh đã được lưu ở đâu
    print(f"\n[INFO] Test plot image saved to: {output_path}")
    assert os.path.exists(output_path)


def test_geometric_solver_error_with_3_variables():
    """Kiểm tra bộ giải hình học báo lỗi khi có hơn 2 biến."""
    problem_3_vars = {
        "objective": {"type": "maximize", "coefficients": [1, 1, 1]},
        "variables": ["x1", "x2", "x3"],
        "constraints": []
    }
    solution, _ = solve_with_geometric_method(problem_3_vars)
    assert solution["status"] == "Error"
    assert "exactly 2 variables" in solution["message"]

