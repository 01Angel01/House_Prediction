from src.utils.logger import default_logger as logger

try:
    logger.info("Running")
except:
    logger.info("Not Running")
