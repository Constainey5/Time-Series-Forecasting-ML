
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_DATA_POINTS = 100
FORECAST_HORIZON = 10

print("
--- Time Series Forecasting with Machine Learning ---")

# --- Generate Synthetic Time Series Data ---
def generate_synthetic_data(n_points):
    print("Generating synthetic time series data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    # Simulate a trend, seasonality, and noise
    trend = np.linspace(0, 20, n_points)
    seasonality = 10 * np.sin(np.linspace(0, 3 * np.pi, n_points))
    noise = np.random.normal(0, 1.5, n_points)
    data = trend + seasonality + noise
    
    df = pd.DataFrame({'Date': dates, 'Value': data})
    df.set_index('Date', inplace=True)
    print("Synthetic data generated.")
    return df

# --- Feature Engineering for Time Series ---
def create_features(df, lag_features=3, rolling_window=3):
    print("Creating time series features...")
    # Lag features
    for i in range(1, lag_features + 1):
        df[f'Lag_{i}' ] = df['Value'].shift(i)
    
    # Rolling mean
    df[f'Rolling_Mean_{rolling_window}' ] = df['Value'].rolling(window=rolling_window).mean().shift(1)
    
    # Time-based features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    df.dropna(inplace=True)
    print("Features created.")
    return df

# --- Train and Forecast with Linear Regression ---
def train_and_forecast(df, forecast_horizon):
    print("Training Linear Regression model and forecasting...")
    # Define features (X) and target (y)
    features = [col for col in df.columns if col != 'Value']
    X = df[features]
    y = df['Value']
    
    # Split data into training and testing sets
    train_size = len(df) - forecast_horizon
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE for Linear Regression: {rmse:.2f}")
    
    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Training Data')
    plt.plot(y_test.index, y_test, label='Actual (Test Data)', color='orange')
    plt.plot(y_test.index, predictions, label='Forecast (Linear Regression)', color='green', linestyle='--')
    plt.title('Time Series Forecasting with Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_results.png')
    plt.close()
    print("Forecast results plotted to forecast_results.png.")
    
    return predictions


if __name__ == "__main__":
    data_df = generate_synthetic_data(NUM_DATA_POINTS)
    featured_df = create_features(data_df)
    forecasted_values = train_and_forecast(featured_df, FORECAST_HORIZON)
    print("Time series forecasting script finished.")
