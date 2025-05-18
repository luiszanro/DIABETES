import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.graphics import record_on_output
from sklearn.preprocessing import StandardScaler


def scale_features(X):
    """
    Scales the features using StandardScaler.
    Args:
        X (pd.DataFrame): The feature set.
    Returns:
        pd.DataFrame: Scaled feature set.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets (80% train, 20% test)
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_scaled


def overview(data, output_dir):
    '''
    Erstelle einen Überblick über einige Eigenschaften der Spalten eines DataFrames.
    VARs
        data: Der zu betrachtende DataFrame
    RETURNS:
        None
    '''
    if data.shape[0] == 0:
        raise ValueError("DataFrame has no rows.")
        
    summary_overview = pd.DataFrame({
        'dtype': data.dtypes,
        'total': data.count(),
        'missing': data.isna().sum(),
        'missing%': data.isna().mean() * 100,
        'n_uniques': data.nunique(),
        'uniques%': data.nunique() / data.shape[0] * 100,
        'uniques': [data[col].unique() for col in data.columns]
    })
    output_dir = record_on_output(output_dir)
    filename = "overview_dataframe.csv"
    #file_path = f"{output_dir}{filename}"
    file_path = os.path.join(output_dir, filename)
    summary_overview.to_csv(file_path, index=True)
    print("overview def is completed")
    #print(summary_overview)
    return file_path
    

def outliers_based_zscores(data):
    df = data
    # Calculate the Z-scores for the BMI column
    df['BMI_Z'] = (df['BMI'] - df['BMI'].mean()) / df['BMI'].std()

    # Identify outliers based on Z-scores (greater than 3 or less than -3)
    outliers = df[(df['BMI_Z'] > 5) | (df['BMI_Z'] < -5)]

    # Display outliers
    print("\nOutliers based on Z-scores in the BMI column:\n{outliers}\n\n")

    return outliers

def exclude(data):
    print("\n\nThe exclude starts: ")
    df = data
    # Define the threshold for outliers (adjustable)
    outliers_threshold = 5

    # Filter the dataframe to exclude outliers
    df_no = df[(df['BMI_Z'] <= outliers_threshold) & (df['BMI_Z'] >= -outliers_threshold)]

    # Verify the shape of the new dataframe
    print(f"\nOriginal dataset size: {df.shape[0]} rows")
    print(f"\nDataset size after removing outliers: {df_no.shape[0]} rows")

    # Display the first few rows of the filtered dataframe
    print(f"\nDisplay the first few rows of the filtered dataframe {df_no.head()}\n\n")
    return df_no

def force_binary(data):
    df = data
    # Step 1: Identify the individuals in Diabetes_012 == 0 who meet the criteria
    df.loc[(df['Diabetes_012'] == 0) & 
        (df[['Overweight', 'HighBP', 'HighChol']].eq(1).all(axis=1)), 'Diabetes_012'] = 1

    # Step 2: Convert all 2s to 1
    df.loc[df['Diabetes_012'] == 2, 'Diabetes_012'] = 1

    # Step 3: Print confirmation of changes
    print("\nUpdated Diabetes_012 column successfully.\n\n\n")
    return df
