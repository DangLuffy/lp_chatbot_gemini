# /tests/test_chatbot_nlp.py

import pytest
from app.chatbot.nlp import extract_lp_from_text

# Sử dụng parametrize để chạy cùng một test với nhiều bộ dữ liệu khác nhau
@pytest.mark.parametrize("text_input, expected_output", [
    # Case 1: Tiếng Việt, Maximize, các biến x1, x2
    (
        "Tối đa hóa 3x1 + 5x2 với điều kiện x1 <= 4, 2x2 <= 12, 3x1 + 2x2 <= 18",
        {
            "objective": {"type": "maximize", "coefficients": [3.0, 5.0]},
            "variables": ["x1", "x2"],
            "constraints": [
                {"name": "c1", "coefficients": [1.0, 0.0], "type": "<=", "rhs": 4.0},
                {"name": "c2", "coefficients": [0.0, 2.0], "type": "<=", "rhs": 12.0},
                {"name": "c3", "coefficients": [3.0, 2.0], "type": "<=", "rhs": 18.0},
            ]
        }
    ),
    # Case 2: Tiếng Anh, Minimize, các biến x, y
    (
        "min 10x - 2y s.t. x+y>=2, 5x <= 25",
        {
            "objective": {"type": "minimize", "coefficients": [10.0, -2.0]},
            "variables": ["x", "y"],
            "constraints": [
                {"name": "c1", "coefficients": [1.0, 1.0], "type": ">=", "rhs": 2.0},
                {"name": "c2", "coefficients": [5.0, 0.0], "type": "<=", "rhs": 25.0},
            ]
        }
    ),
    # Case 3: Dạng viết không có dấu nhân, hệ số âm
    (
        "max -z + 2k subject to: -5z + k == 0",
        {
            "objective": {"type": "maximize", "coefficients": [2.0, -1.0]},
            "variables": ["k", "z"],
            "constraints": [
                {"name": "c1", "coefficients": [1.0, -5.0], "type": "==", "rhs": 0.0},
            ]
        }
    )
])
def test_nlp_extraction(text_input, expected_output):
    """Kiểm tra chức năng trích xuất LP từ văn bản."""
    problem_data, logs = extract_lp_from_text(text_input)
    
    # In logs nếu test thất bại để dễ debug
    print("NLP Logs:", logs)
    
    assert problem_data is not None
    assert problem_data["objective"]["type"] == expected_output["objective"]["type"]
    assert problem_data["variables"] == expected_output["variables"]
    assert problem_data["objective"]["coefficients"] == expected_output["objective"]["coefficients"]
    assert len(problem_data["constraints"]) == len(expected_output["constraints"])
    # So sánh từng ràng buộc (có thể cần logic so sánh phức tạp hơn nếu thứ tự không đảm bảo)
    for i, constr in enumerate(problem_data["constraints"]):
        expected_constr = expected_output["constraints"][i]
        assert constr["coefficients"] == expected_constr["coefficients"]
        assert constr["type"] == expected_constr["type"]
        assert constr["rhs"] == expected_constr["rhs"]

def test_nlp_invalid_text():
    """Kiểm tra NLP với văn bản không hợp lệ."""
    invalid_text = "Xin chào, thời tiết hôm nay thế nào?"
    problem_data, _ = extract_lp_from_text(invalid_text)
    assert problem_data is None

