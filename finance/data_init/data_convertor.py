"""
Convert CSV and JSON to MongoDB Collection format
"""

import os
import shutil
from utils import convert_csv_to_json_lines, convert_json_to_json_lines


PROJECT_RELATIVE_PATH = "."
PROJECT_DATA_PATH = "data"
PROJECT_RELATIVE_DATA_PATH = f"{PROJECT_RELATIVE_PATH}/{PROJECT_DATA_PATH}"

# Input

data_csv = [
    "users_data.csv",
    "cards_data.csv",
    "transactions_data.csv",
]

data_json = ["mcc_codes.json", "train_fraud_labels.json"]


def process():
    # check data dir
    if "data" not in os.listdir(PROJECT_RELATIVE_PATH):
        print("Please, run './data/kaggle_wrapper.sh' first")
        print("Execution Interrupted.")

    # check data/json dir
    elif "json" not in os.listdir(PROJECT_RELATIVE_DATA_PATH):
        os.mkdir(f"{PROJECT_RELATIVE_DATA_PATH}/json")
        convert_csv_to_json_lines(
            data_csv=data_csv, relative_path=PROJECT_RELATIVE_DATA_PATH
        )
        convert_json_to_json_lines(
            data_json=data_json, relative_path=PROJECT_RELATIVE_DATA_PATH
        )
        print(
            "'data/json' folder doesn't exist, 'data/json' folder is created with converted data"
        )
        print("Executed Successfully!")

    # refresh data/json folder
    else:
        shutil.rmtree(f"{PROJECT_RELATIVE_DATA_PATH}/json")
        os.mkdir(f"{PROJECT_RELATIVE_DATA_PATH}/json")
        convert_csv_to_json_lines(
            data_csv=data_csv, relative_path=PROJECT_RELATIVE_DATA_PATH
        )
        convert_json_to_json_lines(
            data_json=data_json, relative_path=PROJECT_RELATIVE_DATA_PATH
        )
        print("'data/json' folder is refreshed by data convertor")
        print("Executed Successfully!")


if __name__ == "__main__":
    process()
