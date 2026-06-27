import logging
import os

def setup_logger():
    """
    Configures a logger that outputs to both the console and a file.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("medical_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if handler already exists
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File Handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "system_log.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger