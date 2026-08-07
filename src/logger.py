import logging
import os
from datetime import datetime

# 1. Extract the current date for the folder name (e.g., '2026-08-07')
DATE_FOLDER = datetime.now().strftime("%Y-%m-%d")

# 2. Extract the current time for the log file name (e.g., '23_19_30.log')
LOG_FILE_NAME = f"{datetime.now().strftime('%H_%M_%S')}.log"

# 3. Define directory path: root_directory/logs/YYYY-MM-DD/
logs_dir_path = os.path.join(os.getcwd(), "logs", DATE_FOLDER)

# 4. Create the date-based directory if it doesn't exist
os.makedirs(logs_dir_path, exist_ok=True)

# 5. Define full log file path: root_directory/logs/YYYY-MM-DD/HH_MM_SS.log
LOG_FILE_PATH = os.path.join(logs_dir_path, LOG_FILE_NAME)

# 6. Configure logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)