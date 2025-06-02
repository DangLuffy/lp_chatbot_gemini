// /static/js/main.js

/**
 * Hàm chính xử lý logic của ứng dụng chat sau khi DOM được tải.
 */
function main() {
    console.log("Hàm main() đang được thực thi. main.js đang hoạt động.");

    const chatForm = document.getElementById('chat-form'); // Giả sử có form với ID 'chat-form'
    const messageInput = document.getElementById('message-input'); // Giả sử có input với ID 'message-input'
    const chatMessagesContainer = document.getElementById('chat-messages'); // Giả sử có div với ID 'chat-messages'

    if (chatForm && messageInput && chatMessagesContainer) {
        chatForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Ngăn form submit theo cách truyền thống

            const userMessageText = messageInput.value.trim();
            if (!userMessageText) {
                return; // Không gửi nếu tin nhắn trống
            }

            // Hiển thị tin nhắn của người dùng
            appendMessageToChat(userMessageText, 'user', chatMessagesContainer);
            messageInput.value = ''; // Xóa input sau khi gửi

            // Hiển thị chỉ báo "Bot đang soạn..." (tùy chọn)
            const typingIndicator = appendMessageToChat('Bot đang soạn...', 'bot', chatMessagesContainer, true);

            try {
                // Tạo FormData để gửi dữ liệu
                const formData = new FormData();
                formData.append('message', userMessageText);

                // Gửi yêu cầu POST đến endpoint của chatbot
                // URL này cần được cấu hình đúng dựa trên router của bạn
                const response = await fetch('/send_message', { // HOẶC /chatbotui/send_message tùy cấu hình
                    method: 'POST',
                    body: formData
                });

                // Xóa chỉ báo "Bot đang soạn..."
                if (typingIndicator && typingIndicator.parentNode) {
                    typingIndicator.parentNode.removeChild(typingIndicator);
                }

                if (!response.ok) {
                    // Xử lý lỗi từ server
                    let errorMsg = `Lỗi từ server: ${response.status} ${response.statusText}`;
                    try {
                        const errorData = await response.json();
                        errorMsg = errorData.detail || errorData.message || errorMsg;
                    } catch (e) {
                        // Không thể parse JSON, dùng thông báo mặc định
                    }
                    appendMessageToChat(errorMsg, 'bot', chatMessagesContainer);
                    console.error('Error sending message:', errorMsg);
                    return;
                }

                const data = await response.json();
                // Hiển thị phản hồi của bot
                appendMessageToChat(data.bot_response, 'bot', chatMessagesContainer);

            } catch (error) {
                // Xử lý lỗi mạng hoặc lỗi khác
                if (typingIndicator && typingIndicator.parentNode) {
                    typingIndicator.parentNode.removeChild(typingIndicator);
                }
                appendMessageToChat('Lỗi kết nối tới server. Vui lòng thử lại.', 'bot', chatMessagesContainer);
                console.error('Network or other error:', error);
            }
        });
    } else {
        console.warn("Một số phần tử chat (form, input, messages container) không được tìm thấy. Chat JS sẽ không hoạt động đầy đủ.");
    }
}

// Hàm này được gọi khi trang HTML được tải hoàn toàn
document.addEventListener('DOMContentLoaded', function () {
    console.log("Trang đã được tải hoàn toàn. Gọi hàm main().");
    main(); // Gọi hàm chính để khởi chạy logic ứng dụng
});

/**
 * Hàm tiện ích để thêm tin nhắn vào giao diện chat.
 * @param {string} text Nội dung tin nhắn.
 * @param {string} sender Người gửi ('user' hoặc 'bot').
 * @param {HTMLElement} container Phần tử DOM chứa các tin nhắn.
 * @param {boolean} isTyping Nếu là true, trả về phần tử DOM của tin nhắn (cho chỉ báo "đang soạn").
 * @returns {HTMLElement|null} Phần tử DOM của tin nhắn nếu isTyping là true, ngược lại là null.
 */
function appendMessageToChat(text, sender, container, isTyping = false) {
    if (!container) return null;

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message'); // Sử dụng class 'message' từ style.css

    if (sender === 'user') {
        messageDiv.classList.add('user-message'); // Sử dụng class 'user-message'
    } else {
        messageDiv.classList.add('bot-message'); // Sử dụng class 'bot-message'
    }

    messageDiv.textContent = text;
    container.appendChild(messageDiv);

    // Tự động cuộn xuống tin nhắn mới nhất
    container.scrollTop = container.scrollHeight;

    return isTyping ? messageDiv : null;
}

// Bạn có thể thêm các hàm JavaScript khác ở đây cho các tương tác khác trên trang web.
// Ví dụ:
// function toggleTheme() {
//     document.body.classList.toggle('dark-theme');
// }
