import os
import sys
import dill
from mlproject.src.exception import CustomException
from mlproject.src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error while saving object: {e}")
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test, y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            logging.info(f"Best parameters for model {list(models.keys())[i]}: {gs.best_params_}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report 
    except Exception as e:
        logging.error(f"Error while evaluating models: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            return obj 
    except Exception as e:
        logging.error(f"Error while loading object: {e}")
        raise CustomException(e, sys)