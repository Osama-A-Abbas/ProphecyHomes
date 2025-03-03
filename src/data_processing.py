# src/data_processing.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file to be loaded.
    
    Returns:
    - DataFrame containing the loaded data.
    """
    
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the dataset (e.g., handle missing values, encode categorical variables, etc.).
    
    Parameters:
    - df: DataFrame, the raw data.
    
    Returns:
    - DataFrame, the preprocessed data.
    """
   
    # Example: Fill missing values 
    df = df.fillna(0)

    # Example: Convert categorical columns to dummy variables
    # df = pd.get_dummies(df)
    
    # Handle categorical variables
    if 'ocean_proximity' in df.columns:
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
        
    # (Optional) Check for any remaining missing values
    if df.isnull().values.any():
        print("Warning: There are still missing values in the dataset after preprocessing.")


    # (Optional) Feature Scaling - Uncomment if needed
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    # df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df