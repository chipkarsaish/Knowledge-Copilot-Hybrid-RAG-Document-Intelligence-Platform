import sys
import os
import logging
from datetime import datetime


# -------- LOGGING SETUP --------
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s'
)
# -------------------------------


def get_error_detail(error):
    _, _, exc_tb = sys.exc_info()
    return (
        f"Error occurred in python script "
        f"[{exc_tb.tb_frame.f_code.co_filename}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error_message = get_error_detail(error)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Division by zero error occurred")
        raise CustomException(e)