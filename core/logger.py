# /core/logger.py

import logging
import sys
from .config import settings # Import settings từ config.py cùng thư mục

def setup_logging():
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng.
    """
    # Lấy root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL) # Đặt mức log từ cấu hình

    # Xóa các handler hiện có để tránh log bị lặp lại nếu hàm này được gọi nhiều lần
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Tạo một StreamHandler để log ra console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL)

    # Tạo formatter và gán nó cho handler
    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)

    # Thêm handler vào root logger
    root_logger.addHandler(console_handler)

    # Log một thông điệp để xác nhận logging đã được thiết lập
    logging.info(f"Logging setup complete. Log level: {settings.LOG_LEVEL}")

    # Ví dụ về cách log từ các module khác:
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.info("This is an info message from my_module.")
    # logger.error("This is an error message from my_module.")

if __name__ == "__main__":
    # Test thử thiết lập logging
    setup_logging()
    
    logger = logging.getLogger("my_test_logger")
    logger.debug("Đây là một thông điệp debug (sẽ không hiển thị nếu LOG_LEVEL là INFO).")
    logger.info("Đây là một thông điệp info.")
    logger.warning("Đây là một thông điệp warning.")
    logger.error("Đây là một thông điệp error.")
    logger.critical("Đây là một thông điệp critical.")

    # Thay đổi log level để xem debug message
    # settings.LOG_LEVEL = "DEBUG" # Điều này không ảnh hưởng đến setup_logging đã chạy
    # print("\nRe-setting up logging with DEBUG level (for demonstration):")
    # # Để thay đổi log level động, bạn cần cấu hình lại logger trực tiếp hoặc gọi lại setup_logging
    # # Ví dụ đơn giản là thay đổi level của root logger:
    # logging.getLogger().setLevel("DEBUG")
    # logger.debug("Bây giờ thông điệp debug này sẽ hiển thị.")
