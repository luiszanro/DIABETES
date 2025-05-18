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

def implementing_voting(X_train, X_test, y_train, y_test, output_dir):
    """
    Apply SMOTE.
    APPLy Voting.
    """
    # Selecting top 10 columns
    selected_columns = [
        'HighChol', 'HighBP', 'Overweight', 'CholCheck', 'GenHlth',
        'HeartDiseaseorAttack', 'Sex', 'Age', 'DiffWalk', 'Stroke'
        ]
    
    #  Ensure only valid columns are selected
    valid_columns = [feat for feat in selected_columns if feat in X_train.columns]
    X_train_selected = X_train[valid_columns]
    X_test_selected = X_test[valid_columns]

    # Print final train selection set to verify
    print("\nFinal train selection used:\n\n", X_train_selected.columns.tolist())

    # Apply SMOTE (Oversampling Minority Class)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    #X_test_df = scaler.transform(X_test_df)
    X_test_selected = scaler.transform(X_test_selected)

    #  Print class distribution
    print("\nClass distribution after SMOTE:\n\n", pd.Series(y_train_balanced).value_counts())

    #  Initialize Models
    cat = CatBoostClassifier(verbose=0)
    lgb = LGBMClassifier(verbose=-1)
    xgb = XGBClassifier()

    model_list = [('cat', cat), ('lgb', lgb), ('xgb', xgb)]
    clf = VotingClassifier(model_list, voting='soft', n_jobs=-1)

    #  Train models with SMOTE-balanced dataset
    clf.fit(X_train_balanced, y_train_balanced)

    #  Prediction using VotingClassifier
    y_probs = clf.predict_proba(X_test_selected)

    #  Added in order to create a threshold which can be adjusted (Adjust classification threshold)
    threshold = 0.3
    y_pred_adjusted = np.where(y_probs[:, 1] >= threshold, 1, 0)

    #  Compute Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs[:, 1])
    auprc = auc(recall, precision)

    #  Plot Precision-Recall Curve
    # plt.plot(recall, precision, marker='.')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve (AUC = {auprc:.2f})')
    # plt.show()
    tun_1, file_path = graph_recall_precision(recall, precision, auprc, output_dir)
    print(f"\n* graph_recall_precision Sucessfully stored at: {file_path} \n\n") 
    print(f"\n Plot Completed = {tun_1}")
    #print(tun_1)
    ################ SMLH + SMS ##############33

    #  Compute & Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_adjusted)    
    cm_img_done = graph_cm(cm)
    print("Completed = ")
    print(cm_img_done)

    #  Generate classification report
    report = classification_report(y_test, y_pred_adjusted, target_names=["No Diabetes (0)", "Diabetes/Risk (1)"])
    print("Performance Metrics:")
    print(report)


    voting_v1 = "yes"
    return X_train_balanced, y_train_balanced, voting_v1