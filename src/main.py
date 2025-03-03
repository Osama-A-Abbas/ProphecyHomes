# src/main.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import load_data, preprocess_data

def explore_data(df):
    """ Perform initial data exploration. """
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Show summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Visualize the distribution of the target variable (e.g., house prices)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['median_house_value'], bins=30, kde=True)
    plt.title('Distribution of Median House Values')
    plt.xlabel('Median House Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Visualize relationships between key variables (e.g., scatter plots)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['median_income'], y=df['median_house_value'])
    plt.title('House Value vs. Median Income')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.show()
    
def train_model(X_train, y_train):
    """ Train a Linear Regression model. """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """ Evaluate the model using test data. """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")


def main():
    # Load the raw data
    file_path = 'data/raw/housing.csv'
    raw_data = load_data(file_path)
    print("Data loaded successfully")

    # Preprocess the data
    processed_data = preprocess_data(raw_data)
    print("Data preprocessed successfully")

    # Explore the data
    explore_data(processed_data)

    # Split the data into features and target variable
    X = processed_data.drop('median_house_value', axis=1)
    y = processed_data['median_house_value']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)
    print("Model trained successfully")

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()