# Stock-model-prediction

Stock Price Prediction Using LSTM and Technical Indicators
Overview
This project involves using a Long Short-Term Memory (LSTM) neural network to predict stock prices, incorporating technical indicators such as Moving Averages (MA) and the Relative Strength Index (RSI).
Data Collection and Preprocessing

    Data Source: The project fetches historical stock data for Apple Inc. (AAPL) from a financial data source.
    Data Range: The data spans from January 2015, with a total of 2014 rows initially collected.
    Features:
        Open, High, Low, Close, Adj Close, and Volume are the primary features.
        Additional features calculated include:
            Moving Averages (MA50 and MA200): 50-day and 200-day moving averages.
            Relative Strength Index (RSI): A momentum indicator measuring the magnitude of recent price changes.

Feature Engineering

    Moving Averages: Calculated for 50 and 200 days to capture long-term and short-term trends.
    RSI: Computed to gauge the stock's recent price changes and identify overbought or oversold conditions.
    Handling NaN Values: After feature engineering, the dataset is cleaned to handle any NaN values, resulting in 1965 rows.

Model Implementation

    LSTM Model:
        The model is built using the Keras library in Python.
        The LSTM layer is used to capture sequential dependencies in the stock price data.
        The model is trained over 100 epochs with a batch size of 48.
        The loss function used is Mean Squared Error (MSE), and the optimizer is Adam.

Training and Evaluation

    Training Process:
        The model is trained on the preprocessed data, with the loss decreasing over epochs, indicating model convergence.
        Example loss values at different epochs:

        text
        Epoch 1/100 - loss: 0.0124
        Epoch 10/100 - loss: 0.0010
        Epoch 50/100 - loss: 3.9973e-04

        The training process is efficient, with each epoch taking approximately 1 second and 14 milliseconds per step.
    Evaluation:
        The model's performance is evaluated based on the MSE loss, which decreases significantly over the training epochs, indicating a good model fit.

Code Structure
The code is organized into several sections:

    Data Fetching: Scripts for fetching historical stock data.
    Data Preprocessing: Code for cleaning and preparing the data by calculating additional features like MA and RSI.
    Model Definition: Code for defining the LSTM model architecture.
    Model Training: Scripts for training the model over the specified number of epochs.
    Evaluation: Code for printing out the loss at each epoch to monitor model performance.
