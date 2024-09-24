import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import numpy as np


class ModelTester:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def evaluate_on_test_data(self, test_arr):
        try:
            logging.info("Loading the trained model")
            # Load the trained model
            model = load_object(self.model_file_path)
            
            # Separate features and labels from the test data
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            # Predict using the trained model
            logging.info("Making predictions on test data")
            predictions = model.predict(X_test)
            
            # Calculate evaluation metrics
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5

            logging.info(f"Test R2 Score: {r2}")
            logging.info(f"Test MAE: {mae}")
            logging.info(f"Test MSE: {mse}")
            logging.info(f"Test RMSE: {rmse}")

            print(f"Test R2 Score: {r2}")
            print(f"Test MAE: {mae}")
            print(f"Test MSE: {mse}")
            print(f"Test RMSE: {rmse}")

            return {"r2_score": r2, "mae": mae, "mse": mse, "rmse": rmse}
        
        except Exception as e:
            raise CustomException(e, sys)

# Usage example
# if __name__ == "__main__":
#     test_data_path = 'path_to_your_test_data.npy'  # Provide the correct path to your test data
#     test_arr = np.load(test_data_path)  # Load your test data here
    
#     tester = ModelTester(model_file_path="artifacts/model.pkl")
#     test_scores = tester.evaluate_on_test_data(test_arr=test_arr)
#     print(test_scores)
