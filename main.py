import os
import argparse
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

import logging as log # Use Python's built-in logging if ElementsKernel isn't installed

from sklearn.model_selection import train_test_split, GridSearchCV

from src.data.load_data import load_data
from src.data.data_preprocessing import scale_features, overview, outliers_based_zscores, exclude, force_binary
from src.utils.resampling import combine_resampling
from src.utils.graphics import plotheatmap, trainer_importance_plot
from src.models.classifier import ModelTrainer
from src.models.evaluation import evaluate_model
from src.models.voting import implementing_voting
from src.models.hyperparameter_tunning import tunning_hyperparam
from src.models.voting2 import voting_with_tunning



def defineSpecificProgramOptions():

    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction Pipeline")
    
    # Explicit definition of the arguments
    data_help = (
        "Choose a dataset which starts with the first column as the most significative."
        "Path to the dataset CSV file"
    )
    model_help = (
        "Choose the model that will be of used for the etc"
    )
    execution_mode_help = (
        "Choose the mode binary 1 or states 2. "
        "1 will offer 1 or 0."
        "2 will offer 0, 1, 2"
    )
    output_help = ("choose a new path if necessary otherwise it will output by default"
                   "in './ouput/'.")

    # Arguments seen by parser

    parser.add_argument('-i', '--data-path', type=str, default='data/raw/diabetes_dataset.csv',
                        help=data_help)    
    parser.add_argument('-m', '--model', type=str, default='LogisticRegression',
                        choices=['RandomForest', 'XGBoost', 'LogisticRegression'],
                        help=model_help,)
    parser.add_argument('-s','--choose-mode', type=str, choices=['1','2'], default="1", #action='store_true',
                        help=execution_mode_help,)


    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default='output/',
        type=str,
        help=output_help,
    )

    # parser.add_argument(
    #     "-c",
    #     "--config_file",
    #     type=str,
    #     nargs=1,
    #     help=config_help,
    # )

    parser.add_argument(
        "--workdir",
        metavar="DIR",
        type=str,
        help="Absolute path to the working directory",
    )

    parser.add_argument(
        "--logdir",
        metavar="DIR",
        type=str,
        help="Relative path to log directory.",
    )
    
    #args = parser.parse_args()
    #return parser.parse_args()
    
    return parser.parse_args()

def main():
    
    print("Hello from Luis!")
    print("The first part of the program will start")
    args = defineSpecificProgramOptions()
    # Initialize logging
    log_file_path = 'output/diabetes_log_output.log'
    log.basicConfig(filename=log_file_path, level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    my_logger = log.getLogger("DIABETES")
    my_logger.info("Entering DIABETES main().")

    # Logging parsed arguments
    my_logger.info(f"Dataset Path: {args.data_path}")
    my_logger.info(f"Model Choice: {args.model}")
    my_logger.info(f"Execution Mode: {args.choose_mode}")
    my_logger.info(f"Output Path: {args.output}")
    my_logger.info(f"Working Directory: {args.workdir}")
    my_logger.info(f"Log Directory: {args.logdir}")
    
    # Simulate execution
    print(f"Loading data from {args.data_path}")
    print(f"Using model: {args.model}")
    print(f"Execution mode: {args.choose_mode}")
    print(f"Results will be saved to: {args.output if args.output else 'Default location'}")
    print(f"\n\n\n###########################################")

    #############
    ## Load data
    df = load_data(args.data_path)
    #############
    ############
    ## EDA
    ############

    store_overview = overview(df, args.output)
    print(f"\n* Overview of the dataframe successfully saved at: {store_overview} \n\n")

    file_path = os.path.join(args.output, "summary_statistics.csv")  # Save CSV 
    df.describe().to_csv(file_path,index=True)
    print(f"* Describe dataframe was successfully saved at: {file_path} \n\n") 

    ############
    ## Preprocessing
    ###########

    # Adding layer overweight
    df['Overweight'] = df['BMI'].apply(lambda x: 1 if x >= 25 else 0)  
    print(f"Adding overweight column: \n\n {df.columns}")  
        
    # Correlation matrix for heatmapplot 
    correlation_matrix, file_path = plotheatmap(df, args.output)
    print(f"\n\n* Plotheatmap Sucessfully saved at: {file_path} \n\n") 
    print(f"Printing the correlation matrix: \n\n {correlation_matrix}")
    
    # Outliers, identying and excluding
    outliers = outliers_based_zscores(df)
    excluders = exclude(df)
#    print(f"\n\n###################{df['Diabetes_012'].nunique()}##############\n\n")
    
    # For debugging it shows the selection mode of the program
    print(f"The selection method is {args.choose_mode} where mode 1 is binary and 2 is states\n\n")

    # Check if first column is managed in states or binary
    if args.choose_mode == 2:
        print("The selection method is 3 states")
        '''
        This is state is on work yet
        The data column Diabetes_012 is managed as states 0, 1 and 2 
        
        print("The column 'Diabetes_012' contains 3 states 0, 1 and 2.")
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
        '''
    else:
        #####################
        # BINARY MODE BELOW
        #####################
        print("############ Binary Mode ############")
        print("\nThe column 'Diabetes_012' contains ONLY 0 and 1.\n\n")
        # Define features (X) and target variable (y)
        df_binary = force_binary(df)
        file_path = os.path.join(args.output, "Binary_dataframe.csv")  # Save CSV 
        df_binary.to_csv(file_path,index=True)
        print(f"* The Binary dataframe was successfully stored at: {file_path} \n\n") 



        # Define features (X) and target variable (y)
        X = df_binary.drop(columns=['Diabetes_012'])  # Exclude the target column
        y = df_binary['Diabetes_012']  # Target column
            
        # Preprocess features (scale)
        X_scaled = scale_features(X)
        
        # Split the dataset into training and testing sets (80% train, 20% test)
        #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        #####################
        # Model [RandomForest', 'XGBoost', 'LogisticRegression']
        # args.model=modelt_type
        # LogisticRegression by default
        #####################
        print("############ MODEL ############")
        # For simplicity we will consider default LogisticRegression 
        model_type = args.model
        if model_type == "RandomForest":
            print("This is work in progress for RandomForest...")
        elif model_type == "XGBoost":
            print("This is work in progress for  XGBoost...")
            # Note: Consider
            # X_scaled, y = combine_resampling(X_scaled, y)
        else:
            #This sections runs with Logistic regression
            print("\nUsing LogisticRegression as the selected model.\n\n")

            # Training model first version to train the model
            
            # Initialize model trainer
            print("\nInitialize Model Trainer \n\n")
            trainer = ModelTrainer(model_type=args.model, class_weight='balanced')
            
            # Train the model
            print("\nTrain the Model \n\n")
            trainer.train(X_train, y_train)

            # Convert X_train to DataFrame to ensure column names are retained
            #X_train_df = pd.DataFrame(X_train, columns=X.columns)  # Ensure X_train is a DataFrame

            #trainer_importance = pd.Series(trainer.coef_[0], index=X_train_df.columns)  # Assign column names
            #trainer_importance = pd.Series(trainer.coef_[0], index=X_train)  # Assign column names
            trainer_importance = pd.Series(trainer.model.coef_[0], index=X_train.columns)  # Assign column names

            # Print Selection of the best 10 columns
            print("\nTop Features (Logistic Regression):\n\n")
            print(trainer_importance.sort_values(ascending=False).head(10))
            # Sort features by importance
            trainer_importance_sorted = trainer_importance.sort_values(ascending=False)
            
            # Plot Image with colors
            # Farben aus Viridis generieren and more config for plot or make them on graphics.py           
            #trainer_importance_sorted.head(10).plot(kind='bar', color='colors')
            topten, file_path = trainer_importance_plot(trainer_importance_sorted, args.output)
            print(f"\n\n* trainer_importance_sorted Sucessfully saved at: {file_path} \n\n")
            print(f"Saved completed: {topten} \n\n")

            ############
            ## Voting
            ###########
            print("############ VOTING ############\n\n")
            print("Start voting!\n\n")

            # This section starts SMOTE and uses VotingClassifier
            X_train_balanced, y_train_balanced, vot_1 = implementing_voting(X_train, X_test, y_train, y_test, args.output)
            
            print("\nCompleted = ")
            print(vot_1)
            
            ############
            ## Tunning Hyperparameter
            ###########
            print("############ Tunning hyperparameter ############\n\n")
            print("Start Tunning hyperparameter!\n\n")
            
            # Grid hyperparam and searchCV per model
            tun_1 = tunning_hyperparam(X_train_balanced, y_train_balanced)
            print("\nCompleted = ")
            print(tun_1)
            ############
            ## Voting with some additions
            ###########
            print("############ VOTING ############\n\n")
            print("\nStart voting with some additional tunning!\n\n")

            # This section starts SMOTE and uses VotingClassifier and extreme tunnings
            X_train_balanced, y_train_balanced, vot_2 = voting_with_tunning(X_train, X_test, y_train, y_test)
            print("\nCompleted = ")
            print(vot_2)
            print()
        # #### NOTE EVerything below could be used for states
        # # Handle class imbalance using resampling
        # X_resampled, y_resampled = combine_resampling(X_scaled, y)
        
        # # Initialize model trainer
        # trainer = ModelTrainer(model_type='RandomForest')
        
        # # Train the model
        # trainer.train(X_resampled, y_resampled)
        
        # # Make predictions (example using test data)
        # y_pred = trainer.predict(X_resampled)  # In practice, you'd use X_test
        
        # # Evaluate the model
        # evaluate_model(y_resampled, y_pred)

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


    # """            
    # ###################### with ARGS
    # This section is intended to be added once we complete all args

    # # Optionally resample to balance classes
    # #args.model
    # if args.use_smote:
    #     print("[INFO] Applying SMOTE and undersampling...")
    #     X_scaled, y = combine_resampling(X_scaled, y)
    
    # # Initialize and train model
    # print(f"[INFO] Training model: {args.model}")
    # trainer = ModelTrainer(model_type=args.model)
    # trainer.train(X_scaled, y)
    
    # # Predict on training data (for testing — in practice, use test set)
    # y_pred = trainer.predict(X_scaled)

    # # Evaluate
    # print("[INFO] Evaluating model...")
    # evaluate_model(y, y_pred)
    # ##################### with ARGS
    # """ 