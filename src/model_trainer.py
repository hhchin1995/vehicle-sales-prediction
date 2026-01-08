# Model Trainer Class
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow mlflow to log all params, metrics and etc for a model
mlflow.autolog()
# Save experiments in <project root>/mlruns
mlflow.set_tracking_uri("../mlruns")  

mlflow.set_experiment("Vehicle_Price_Prediction")

class ModelTrainer:
    def __init__(self, name, model, X_train, y_train, X_test, y_test):
        self.name = name
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds_test = self.model.predict(self.X_test)
        true_test = self.y_test
        preds_train = self.model.predict(self.X_train)
        true_train= self.y_train
        return {
            # 'MAE': mean_absolute_error(true, preds),
            'RMSE': mean_squared_error(true_test, preds_test),
            'R2 Test': r2_score(true_test, preds_test),
            'R2 Train': r2_score(true_train, preds_train)
        }

    def run(self):
        metrics = {}
        try:
            with mlflow.start_run(run_name=self.name):
                self.train()
                metrics = self.evaluate()
                for k, v in metrics.items():
                    mlflow.log_metric(k.lower(), float(v))
                mlflow.log_param("model", self.name)
                return metrics
        except Exception as e:
            print(f"MLflow run failed: {e}")
            raise
