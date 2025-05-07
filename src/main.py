import argparse
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from src.data.load_data import load_data
from src.data.data_preprocessing import scale_features, overview, outliers_based_zscores, exclude, force_binary
from src.utils.resampling import combine_resampling
from src.utils.graphics import plotheatmap, trainer_importance_plot
from src.models.classifier import ModelTrainer
from src.models.evaluation import evaluate_model



def define_specific_program_options():
    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction Pipeline")
    
    parser.add_argument('--model', type=str, default='RandomForest',
                        choices=['RandomForest', 'XGBoost', 'LogisticRegression'],
                        help='Which model to use')
    
    parser.add_argument('--data-path', type=str, default='diabetes_data.csv',
                        help='Path to the dataset CSV file')

    parser.add_argument('--use-smote', action='store_true',
                        help='Apply SMOTE oversampling')

    #args = parser.parse_args()
    return parser.parse_args()

def main():
    
    print("Hello from diabetes!")
    
    ###################### with ARGS
    #Use Args.model, args.data_path
    args = define_specific_program_options()
    
    print(f"[INFO] Loading data from: {args.data_path}")
    df = load_data(args.data_path)
    ###################### with ARGS
    
    # Load data
    #df = load_data('data/raw/diabetes_data.csv')
    #df = load_data('data\raw\diabetes_dataset.csv')
    df = load_data(r'data\raw\diabetes_dataset.csv')
    
    # EDA
    overview(df)
    df.describe()
    
    # Preprocessing
    
    #Adding layer overweight
    df['Overweight'] = df['BMI'].apply(lambda x: 1 if x >= 25 else 0)  
    print(df.columns)  
        
    # Correlation matrix for heatmapplot 
    correlation_matrix = plotheatmap(df)
    print(correlation_matrix)
    
    # outliers, identying and excluding
    outliers = outliers_based_zscores(df)
    exclude = exclude(df)
    
    # Check if states or binary
    if df['Diabetes_012'].isin([0, 1, 2]).all():
        print("The column 'Diabetes_012' contains 3 states 0, 1 and 2.")
        #######################Review
        #if df_mod['Diabetes_012'].
        
        # convert 3 states 0.0, 1.0, 2.0 to binary
        # Count occurrences of each value in the Diabetes_012 column
        # binary_data = force_binary(df)
        # counts = df['Diabetes_012'].value_counts()
        # Print results
        # print("Count of 0s and 1s in Diabetes_012 column:")
        # print(counts)
        
        # Separate features and target
        #X = df.drop('state', axis=1)
        #y = df['state']
        #########################
        # Define features (X) and target variable (y)
        X = df.drop(columns=['Diabetes_012'])  # Exclude the target column
        y = df['Diabetes_012']  # Target column
            
        # Preprocess features (scale)
        X_scaled = scale_features(X)
        
        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    else:
        ##############################3
        # BINARY MODE BELOW
        print("The column 'Diabetes_012' contains ONLY 0 and 1.")
        # Define features (X) and target variable (y)
        df_binary = force_binary(df)
        # Define features (X) and target variable (y)
        X = df_binary.drop(columns=['Diabetes_012'])  # Exclude the target column
        y = df_binary['Diabetes_012']  # Target column
            
        # Preprocess features (scale)
        X_scaled = scale_features(X)
        
        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            
        ###################### with ARGS
        # Optionally resample to balance classes
        #args.model
        if args.use_smote:
            print("[INFO] Applying SMOTE and undersampling...")
            X_scaled, y = combine_resampling(X_scaled, y)
        
        # Initialize and train model
        print(f"[INFO] Training model: {args.model}")
        trainer = ModelTrainer(model_type=args.model)
        trainer.train(X_scaled, y)
        
        # Predict on training data (for testing — in practice, use test set)
        y_pred = trainer.predict(X_scaled)

        # Evaluate
        print("[INFO] Evaluating model...")
        evaluate_model(y, y_pred)
        ##################### with ARGS
        
        ##################### the code below will be designed considering default
        # args.model=modelt_type
        # Decided for default model_type = [RandomForest', 'XGBoost', 'LogisticRegression'] 
        model_type = "LogisticRegression"
        if model_type == "RandomForest":
            print("This is work in progress...")
        elif model_type == "XGBoost":
            print("This is work in progress...")
            # Note: Consider
            # X_scaled, y = combine_resampling(X_scaled, y)
        else:
            #The code runs by default with LogisticRegression            
            ## training model first version to train the model
            # Initialize model trainer
            trainer = ModelTrainer(model_type='LogisticRegression', class_weight='balanced')
            # Train the model
            trainer.train(X_train, y_train)

            # COMMA 05/05/2025
            
            # Convert X_train to DataFrame to ensure column names are retained
            #X_train_df = pd.DataFrame(X_train, columns=X.columns)  # Ensure X_train is a DataFrame

            #trainer_importance = pd.Series(trainer.coef_[0], index=X_train_df.columns)  # Assign column names
            #trainer_importance = pd.Series(trainer.coef_[0], index=X_train)  # Assign column names
            trainer_importance = pd.Series(trainer.coef_[0], index=X_train.columns)  # Assign column names

            # Print Top Features
            print("Top Features (Logistic Regression):")
            print(trainer_importance.sort_values(ascending=False).head(10))
            # Sort features by importance
            trainer_importance_sorted = trainer_importance.sort_values(ascending=False)
            
            # Farben aus Viridis generieren and more config for plot or make them on graphics.py
           
            #trainer_importance_sorted.head(10).plot(kind='bar', color='colors')
            topten = trainer_importance_plot(trainer_importance_sorted)
            ##use try: just for now 
            if topten != "yes":
                print("Trainer_importance_plot Not ploted")
                
            print("Starting the voting")
            # Perk
            selected_features = [
                'HighChol', 'HighBP', 'Overweight', 'CholCheck', 'GenHlth',
                'HeartDiseaseorAttack', 'Sex', 'Age', 'DiffWalk', 'Stroke'
                ]
            
            #  Ensure only valid features are selected
            valid_features = [feat for feat in selected_features if feat in X_train.columns]
            X_train_selected = X_train[valid_features]
            X_test_selected = X_test[valid_features]
            # Apply SMOTE (Oversampling Minority Class)
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

            scaler = StandardScaler()
            X_train_balanced = scaler.fit_transform(X_train_balanced)
            #X_test_df = scaler.transform(X_test_df)
            X_test_selected = scaler.transform(X_test_selected)

            #  Print class distribution
            print("Class distribution after SMOTE:", pd.Series(y_train_balanced).value_counts())

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
            plt.plot(recall, precision, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (AUC = {auprc:.2f})')
            plt.show()

            #  Generate classification report
            report = classification_report(y_test, y_pred_adjusted, target_names=["No Diabetes (0)", "Diabetes/Risk (1)"])
            print("Performance Metrics:")
            print(report)




            
            
            
            

    
    #### NOTE CHANGE BELOW
    # Handle class imbalance using resampling
    X_resampled, y_resampled = combine_resampling(X_scaled, y)
    
    # Initialize model trainer
    trainer = ModelTrainer(model_type='RandomForest')
    
    # Train the model
    trainer.train(X_resampled, y_resampled)
    
    # Make predictions (example using test data)
    y_pred = trainer.predict(X_resampled)  # In practice, you'd use X_test
    
    # Evaluate the model
    evaluate_model(y_resampled, y_pred)

if __name__ == "__main__":
    main()

############################

# import pandas as pd
# from sklearn.metrics import classification_report

# results = []

# def evaluate_model(y_true, y_pred, model_name='', smote_used=False):
#     report = classification_report(y_true, y_pred, output_dict=True)
#     result_row = {
#         'Model': model_name,
#         'SMOTE': smote_used,
#         'F1_macro': report['macro avg']['f1-score'],
#         'F1_weighted': report['weighted avg']['f1-score'],
#         'Recall_diabetic': report['2.0']['recall'],
#         'Recall_at_risk': report['1.0']['recall']
#     }
#     results.append(result_row)

# # Then at the end of experiment_runner.py:
# def main():
#     for exp in EXPERIMENTS:
#         run_experiment(model_type=exp["model"], use_smote=exp["use_smote"])

#     # Save to CSV
#     df_results = pd.DataFrame(results)
#     df_results.to_csv("experiment_results.csv", index=False)
#     print("\n[INFO] Results saved to experiment_results.csv")

############################