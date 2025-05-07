from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using classification report and ROC-AUC.
    Args:
        y_true (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("ROC-AUC Score:")
    print(roc_auc_score(y_true, y_pred, multi_class='ovr'))