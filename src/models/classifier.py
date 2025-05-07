from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def _init_(self, model_type='RandomForest', class_weight='balanced'):
        """
        Initialize the model trainer with the desired model.
        Args:
            model_type (str): Type of model ('RandomForest', 'XGBoost', 'LogisticRegression').
            class_weight (str): Class weight setting ('balanced' or None).
        """
        print("This is the model type = ")
        print(model_type)
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(class_weight=class_weight, random_state=42)
        elif model_type == 'XGBoost':
            self.model = xgb.XGBClassifier(scale_pos_weight=10, random_state=42)
        elif model_type == 'LogisticRegression':
            #self.model = LogisticRegression(class_weight=class_weight, random_state=42)
            self.model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
        else:
            raise ValueError("Invalid model type. Choose from 'RandomForest', 'XGBoost', 'LogisticRegression'.")
    
    def train(self, X_train, y_train):
        """
        Train the model on the given training data.
        Args:
            X_train (pd.DataFrame): The training feature set.
            y_train (pd.Series): The training target variable.
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        Args:
            X_test (pd.DataFrame): The test feature set.
        Returns:
            pd.Series: The model's predictions.
        """
        return self.model.predict(X_test)
