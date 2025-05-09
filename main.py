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
from sklearn.linear_model import LogisticRegression



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

    parser.add_argument('-i', '--data-path', type=str, default=r'data/raw/diabetes_dataset.csv',
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

    #############
    ## Load data
    df = load_data(args.data_path)
    print("antes de")
    print(df)
    #############
    ############
    ## EDA
    overview(df)
    df.describe()
    ############
    ## Preprocessing
    ###########
    # Adding layer overweight
    df['Overweight'] = df['BMI'].apply(lambda x: 1 if x >= 25 else 0)  
    print(df.columns)  
        
    # Correlation matrix for heatmapplot 
    correlation_matrix = plotheatmap(df)
    print(correlation_matrix)
    
    # Outliers, identying and excluding
    outliers = outliers_based_zscores(df)
    excluded = exclude(df)
    # Check if first column is managed in states or binary
    
    if df['Diabetes_012'].nunique() <= 2:
        print("No Binary")  
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
        ##############################3
        # BINARY MODE BELOW
        print("The column 'Diabetes_012' contains ONLY zeros and ones.")
        # Force Binary
        df_binary = force_binary(df)
        # Define (X) and target variable (y)
        X = df_binary.drop(columns=['Diabetes_012'])  # Exclude the target column
        y = df_binary['Diabetes_012']  # Target column
        # Preprocess Scale
        X_scaled = scale_features(X)
        #print(f"this is X_scaled = {X_scaled}.")
        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        #print(f"this is X_train = {X_train}.")
        #print(X_train.shape, y_train.shape)
        #print(X_test.shape, y_test.shape)
        
        ##################### 
        # The code below is designed with model_type = [RandomForest', 'XGBoost', 'LogisticRegression']
        # For simplicity we will consider default LogisticRegression
        # args.model=modelt_type
        #####################
        
        model_type = "LogisticRegression"
        if model_type == "RandomForest":
            print("This is work in progress for RandomForest...")
        elif model_type == "XGBoost":
            print("This is work in progress for  XGBoost...")
            # Note: Consider to use 
            # X_scaled, y = combine_resampling(X_scaled, y)
        else:
            #####################
            #This sections runs with Logistic regression
            #####################
            print("Using LogisticRegression...")           
            # Training model to get the top 10 significatives 
            # Initialize model trainer with args or without
            # trainer = ModelTrainer(model_type=args.model)
            trainer = ModelTrainer(model_type='LogisticRegression', class_weight='balanced')
            print(f"xxxxxxxxxxxxxxxxxxxxxxxxx>>>\n this is trainer = {trainer}.\n")
            print(trainer.model)
            #trainer = trainer.model
            #print(trainer)
            
            # Train the model
            trainer.train(X_train, y_train)
            
            # Convert X_train to DataFrame to ensure column names are retained
            #X_train_df = pd.DataFrame(X_train, columns=X.columns)  # Ensure X_train is a DataFrame    X
            #X_train_df = pd.DataFrame(X_train)    X
            
            #print(X_train_df.columns)
            #trainer_importance = pd.Series(trainer.model.coef_[0], index=X_train_df.columns)  # Assign column names X
            #trainer_importance = pd.Series(trainer.coef_[0], index=X_train)  # Assign column names X
            trainer_importance = pd.Series(trainer.coef_[0], index=X_train.columns)  # Assign column names

            # Print Selection of the best 10 columns
            print("Top Features (Logistic Regression):")
            print(trainer_importance.sort_values(ascending=False).head(10))
            # Sort features by importance
            trainer_importance_sorted = trainer_importance.sort_values(ascending=False)
            
            # Plot the image with colors
            # Farben aus Viridis generieren and more config for plot or make them on graphics.py           
            #trainer_importance_sorted.head(10).plot(kind='bar', color='colors')
            topten = trainer_importance_plot(trainer_importance_sorted)
            ##use try instead for later quick solution 
            if topten != "yes":
                print("Trainer_importance_plot Not ploted")
            ############
            ## Voting
            ###########
            print("Start voting")
            # This section starts SMOTE and uses VotingClassifier
            X_train_balanced, y_train_balanced, vot_1 = implementing_voting(X_train, X_test, y_train, y_test)
            print("Completed = ")
            print(vot_1)
            ############
            ## Tunning Hyperparameter
            ###########
            print("Tunning hyperparameter")
            # Grid hyperparam and searchCV per model
            tun_1 = tunning_hyperparam(X_train_balanced, y_train_balanced)
            print("Completed = ")
            print(tun_1)
            ############
            ## Voting with some additions
            ###########
            print("Starts voting with some additional tunning")
            # This section starts SMOTE and uses VotingClassifier and extreme tunnings
            X_train_balanced, y_train_balanced, vot_2 = implementing_voting(X_train, X_test, y_train, y_test)
            print("Completed = ")
            print(vot_2)


if __name__ == "__main__":
    main()


#     # Save to CSV
#     df_results = pd.DataFrame(results)
#     df_results.to_csv("experiment_results.csv", index=False)
#     print("\n[INFO] Results saved to experiment_results.csv")

############################