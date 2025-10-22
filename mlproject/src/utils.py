import os
import sys
import dill
from mlproject.src.exception import CustomException
from mlproject.src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # ✅ Correct usage

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error while saving object: {e}")
        raise CustomException(e, sys)
