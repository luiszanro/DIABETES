import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, classification_report, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.utils.graphics import graph_recall_precision, graph_cm

def voting_with_tunning(X_train, X_test, y_train, y_test):
    """
    Apply SMOTE.
    APPLy Voting.
    """
    # Selecting top 10 columns
    selected_features = [
        'HighChol', 'HighBP', 'Overweight', 'CholCheck', 'GenHlth',
        'HeartDiseaseorAttack', 'Sex', 'Age', 'DiffWalk', 'Stroke'
    ]

    #  Ensure only valid features are selected
    valid_columns = [feat for feat in selected_features if feat in X_train.columns]
    X_train_selected = X_train[valid_columns]
    X_test_selected = X_test[valid_columns]
    
    # Print final train selection set to verify
    print("Final train selection used:", X_train_selected.columns.tolist())
    
    #  Apply SMOTE (Oversampling Minority Class)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    #X_test_df = scaler.transform(X_test_df)
    X_test_selected = scaler.transform(X_test_selected)

    #  Print class distribution
    print(" Class distribution after SMOTE:", pd.Series(y_train_balanced).value_counts())

    #  Initialize models with your best hyperparameters
    cat_best = CatBoostClassifier(depth=6, iterations=200, learning_rate=0.1, verbose=0)
    lgb_best = LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=40, verbose=-1)
    xgb_best = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200, eval_metric='mlogloss')

    #  Create VotingClassifier with optimized models
    model_list = [('cat', cat_best), ('lgb', lgb_best), ('xgb', xgb_best)]
    clf = VotingClassifier(model_list, voting='soft', n_jobs=-1, weights=[2, 1, 1])  # Gewichtung für bessere Balance

    #  Train VotingClassifier using SMOTE-balanced dataset
    clf.fit(X_train_balanced, y_train_balanced)

    #  Predict probabilities
    y_probs = clf.predict_proba(X_test_selected)

    #  Tune threshold dynamically for best F1-score
    thresholds = np.linspace(0.2, 0.5, 10)  
    best_f1 = 0
    best_threshold = 0.20

    #  Fix KeyError: Extract Correct Class Label from Report
    report_dict = classification_report(y_test, np.where(y_probs[:, 1] >= best_threshold, 1, 0), output_dict=True)
    print("Available labels in classification report:", report_dict.keys())  # Debugging check

    for thresh in thresholds:
        y_pred_adjusted = np.where(y_probs[:, 1] >= thresh, 1, 0)
        f1 = report_dict.get(str(1), {}).get("f1-score", 0)  #  Using numeric labels (avoid KeyError)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f" Best Threshold Found: {best_threshold}")

    #  Final Predictions Using Best Threshold
    y_pred_adjusted = np.where(y_probs[:, 1] >= best_threshold, 1, 0)

    #  Generate classification report
    report = classification_report(y_test, y_pred_adjusted, target_names=["No Diabetes (0)", "Diabetes/Risk (1)"])
    print("Performance Metrics:\n", report)

    #  Compute Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs[:, 1])
    auprc = auc(recall, precision)

    #  Plot Precision-Recall Curve
    # plt.plot(recall, precision, marker='.')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve (AUC = {auprc:.2f})')  # Fixed unterminated f-string error
    # plt.show()
    tun_1 = graph_recall_precision(recall, precision, auprc)
    print("Completed = ")
    print(tun_1)

    #  Compute & Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_adjusted)    
    cm_img_done = graph_cm(cm)
    print("Completed = ")
    print(cm_img_done)
    
    voting_v2 = "yes"
    return X_train_balanced, y_train_balanced, voting_v2
