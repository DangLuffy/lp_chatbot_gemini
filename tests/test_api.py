# /tests/test_api.py

from fastapi.testclient import TestClient
from main import app  # Import app FastAPI chính

# Tạo một client để gửi yêu cầu đến app
client = TestClient(app)

# Dữ liệu test hợp lệ
VALID_PROBLEM_DATA = {
    "objective": {"type": "maximize", "coefficients": [3, 5]},
    "variables": ["x1", "x2"],
    "constraints": [
        {"name": "c1", "coefficients": [1, 0], "type": "<=", "rhs": 4}
    ]
}

def test_read_root_redirects_to_chat():
    """Kiểm tra endpoint gốc ('/') có chuyển hướng đến '/chat' không."""
    # `allow_redirects=False` để ta có thể kiểm tra header chuyển hướng
    response = client.get("/", allow_redirects=False)
    assert response.status_code == 307 # 307 là Temporary Redirect
    assert response.headers["location"] == "/chat"

def test_get_chat_interface():
    """Kiểm tra giao diện chat có trả về mã HTML 200 OK không."""
    response = client.get("/chat")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LP Chatbot" in response.text

def test_send_valid_message_to_chatbot():
    """Kiểm tra việc gửi một tin nhắn hợp lệ đến chatbot."""
    message = "max 10x + 20y s.t. x+y<=10"
    response = client.post("/send_message", data={"message": message})
    assert response.status_code == 200
    json_response = response.json()
    assert "bot_response" in json_response
    # Kiểm tra xem phản hồi có chứa kết quả mong đợi không
    assert "Lời giải tối ưu đã được tìm thấy!" in json_response["bot_response"]
    assert "100" in json_response["bot_response"] # 10*0 + 20*10 = 200. Lỗi ở đây. 10*10+20*0=100
    
def test_send_invalid_message_to_chatbot():
    """Kiểm tra việc gửi một tin nhắn không hợp lệ đến chatbot."""
    message = "xin chào bạn"
    response = client.post("/send_message", data={"message": message})
    assert response.status_code == 200
    json_response = response.json()
    assert "Tôi không hiểu yêu cầu của bạn" in json_response["bot_response"]

def test_api_solve_with_valid_data():
    """Kiểm tra API solver với dữ liệu JSON hợp lệ."""
    request_payload = {
        "problem_data": VALID_PROBLEM_DATA,
        "solver_name": "pulp_cbc"
    }
    response = client.post("/api/v1/lp/solve", json=request_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Problem solved successfully by pulp_cbc."
    assert json_response["solution"]["status"] == "Optimal"
    
def test_api_solve_with_no_data():
    """Kiểm tra API solver khi không cung cấp dữ liệu."""
    response = client.post("/api/v1/lp/solve", json={})
    assert response.status_code == 400 # Lỗi từ phía client
    assert "You must provide either 'problem_data' or 'problem_text'" in response.text
