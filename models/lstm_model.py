# This code is heavily inspired by the article found here: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# The only claim to authorship is to the use case for which it is now being applied
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from math import sqrt
#from matplotlib import pyplot
import numpy as np
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# fit an LSTM network to training data
def fit_lstm(X, y, batch_size, nb_epoch, neurons):
    keras.backend.clear_session()
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# takes in train and test dataframes with appropriate columns
# fits the lstm model and makes predictions on the test set
# returns both the betting strategy and the predicted change in price
def run_lstm_model(train, test, epochs=10, neurons=3):
    X, y = train.drop(columns=['y']).values, train.y.values

    X_test, y_test = test.drop(columns=['y']).values, test.y.values

    scaler_X = MinMaxScaler(feature_range=(-1,1))
    scaler_y = MinMaxScaler(feature_range=(-1,1))

    # scales all features to be between -1 and 1
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1,)

    
    X_test = scaler_X.transform(X_test)

    model = fit_lstm(X, y, 1, epochs, neurons)

    # forecast the entire training dataset to build up state for forecasting
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    model.predict(X_reshaped, batch_size=1)
    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(X_test.shape[0]):
        # make one-step forecast
        yhat = forecast_lstm(model, 1, X_test[i,:])
        # invert scaling
        yhat = scaler_y.inverse_transform(np.array([yhat]).reshape(1,1))[0,0]
        
        # store forecast
        predictions.append(yhat)

        #print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, actual))

    # report performance
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print('Test RMSE: %.3f' % rmse)

    return predictions, predictions/np.abs(predictions)
    # line plot of observed vs predicted
    #pyplot.plot(raw_values[-12:])
    #pyplot.plot(predictions)
    #pyplot.show()
