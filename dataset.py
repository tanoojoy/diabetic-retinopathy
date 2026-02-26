import os
from kaggle.api.kaggle_api_extended import KaggleApi

current_directory = os.getcwd()
os.environ["KAGGLE_CONFIG_DIR"] = current_directory

kaggleAPI = KaggleApi()
kaggleAPI.dataset_download_files(
    "himeshghoora/iet-codefest-2025",
    path=current_directory,
    unzip=True
)

print('Data source import complete.')
