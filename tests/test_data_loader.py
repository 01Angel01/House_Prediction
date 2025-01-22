# from src.data.data_loader import DataLoader
# from src.utils.logger import default_logger as logger

# if __name__ == "__main__":
#     try:
#         logger.info("Inisialisasi Class")
#         data_loader = DataLoader()

#         logger.info("Start Download Data")
#         data_loader.load_data()
#         logger.info("Data Loaded")
#     except:
#         logger.info("Gagal Load")


from src.data.data_loader import DataLoader
from src.utils.logger import default_logger as logger

if __name__ == "__main__":
    try:
        logger.info("Inisialisasi Class")
        data_loader = DataLoader()
        logger.info(f"Data path resolved: {data_loader.data_path}")

        logger.info("Start Download Data")
        df = data_loader.load_data()
        logger.info(f"Data Loaded: {df.head()}")
    except Exception as e:
        logger.error(f"Gagal Load: {e}")
