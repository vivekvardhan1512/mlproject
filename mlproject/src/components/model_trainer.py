import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from mlproject.src.exception import CustomException
from mlproject.src.logger import logging 

from mlproject.src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Train and Test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'Gradient Boosting': GradientBoostingRegressor(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'XGBClassifier': XGBRegressor()
            }

            model_report : dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]   

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found')                                                    
            logging.info(f'Best Model found on both Training and Test datasets: {best_model_name}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            re_square = r2_score(y_test, predicted)
            logging.info(f'R2 Score of the best model: {re_square}')
            return r2_score
        except Exception as e:
            raise CustomException(e, sys)
    
