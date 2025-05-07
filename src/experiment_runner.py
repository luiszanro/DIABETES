import pandas as pd
from sklearn.metrics import classification_report
from src.data.load_data import load_data
from src.data.data_preprocessing import scale_features
from src.utils.resampling import combine_resampling
from src.models.classifier import ModelTrainer
from src.models.evaluation import evaluate_model
from sklearn.metrics import classification_report

results = []

# Define models and settings to test
EXPERIMENTS = [
    {"model": "RandomForest", "use_smote": False},
    {"model": "RandomForest", "use_smote": True},
    {"model": "XGBoost", "use_smote": False},
    {"model": "XGBoost", "use_smote": True},
    {"model": "LogisticRegression", "use_smote": True}
]

def run_experiment(model_type, use_smote, data_path="data/diabetes_data.csv"):
    print(f"\n[RUNNING] Model: {model_type} | SMOTE: {use_smote}")
    
    # Load and preprocess data
    df = load_data(data_path)
    X = df.drop('state', axis=1)
    y = df['state']
    X_scaled = scale_features(X)
    
    # Optional resampling
    if use_smote:
        X_scaled, y = combine_resampling(X_scaled, y)

    # Train model
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(X_scaled, y)
    
    # Evaluate (on training for now; later use train/test split)
    y_pred = trainer.predict(X_scaled)
    evaluate_model(y, y_pred)
    
    


def evaluate_model(y_true, y_pred, model_name='', smote_used=False):
    report = classification_report(y_true, y_pred, output_dict=True)
    result_row = {
        'Model': model_name,
        'SMOTE': smote_used,
        'F1_macro': report['macro avg']['f1-score'],
        'F1_weighted': report['weighted avg']['f1-score'],
        'Recall_diabetic': report['2.0']['recall'],
        'Recall_at_risk': report['1.0']['recall']
    }
    results.append(result_row)

def main():
    for exp in EXPERIMENTS:
        run_experiment(model_type=exp["model"], use_smote=exp["use_smote"])
        
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results.csv", index=False)
    print("\n[INFO] Results saved to experiment_results.csv")

if __name__ == '__main__':
   main()
