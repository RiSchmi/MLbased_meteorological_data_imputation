# the subsequent code is adapted from the implimention by Brownlee, J. (2020). Multi-step time series forecasting with machine learning for electricity usage. (see readme)
from keras import Sequential, Input
from keras.layers import LSTM, Dense
import numpy as np

# function to reshape 
class LSTM_model():
	def __init__(self, df, timesteps, epochs, batch_size):
		self.df = df
		self.timesteps = timesteps
		self.epochs = epochs, 
		self.batch_size= batch_size
		
	def reshape_data(self, data):
		train = np.array(np.split(data, len(data)/self.timesteps), dtype= 'float')
		train = train.reshape(train.shape[0], train.shape[1], 1)
		return train


	def to_supervised(self, train, n_out):
		# flatten data
		data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
		X, y = list(), list()
		in_start = 0
		# step over the entire history one time step at a time
		for _ in range(len(data)):
			# define the end of the input sequence
			in_end = in_start + self.timesteps
			out_end = in_end + n_out
			# ensure we have enough data for this instance
			if out_end <= len(data):
				x_input = data[in_start:in_end, 0]
				x_input = x_input.reshape((len(x_input), 1))
				X.append(x_input)
				y.append(data[in_end:out_end, 0])
			# move along one time step
			in_start += 1
			
		return np.array(X, dtype= 'float'), np.array(y, dtype= 'float')


	def build_model_lstm(self, train):
		# prepare data
		train_x, train_y = self.to_supervised(train, n_out = self.timesteps)
		# define parameters
		n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
		# define model
		model = Sequential()
		model.add(Input(shape=(n_timesteps, n_features)))
		model.add(LSTM(200, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_outputs))
		model.compile(loss='mae', optimizer='adam')
		
		model.fit(x = train_x, y= train_y, epochs= 5, batch_size= 32, verbose= 0, validation_split= 0.2)
		return model


	def train_uni_LSTM(self, feature):
		train = self.reshape_data(self.df[feature].values)
		model = self.build_model_lstm(train)
		print(f'finish training lstm for {feature}')
		return model

class lstm_impute():
    def __init__(self, df, features, lstm_model):
        self.df = df
        self.features = features
        self.lstm_model = lstm_model


    def impute_with_LSTM(self, index, model, feature):
        
        # define and reshape input data (previous 24 steps)
        input_data = np.array(self.df[feature][index[0] - 24:index[0]], dtype= 'float').reshape(1,24,1)
        # predict based on pretrained model & select only relevant number
        predicted = (model.predict(input_data, verbose=0).tolist()[0])[:len(index)]
        # place values in dataframe
        for n in range(len(index)):
            self.df.loc[index[n], feature] = round(predicted[n],2)
       
    def getitem(self):

        for feature in self.features:
            # train model on feature and training data
            model = self.lstm_model.train_uni_LSTM(feature = feature) 
            current_df = self.df[self.df[feature].isna()].filter(['time_step']).reset_index()

            # iterate over missing data
            index = [current_df.iloc[0]['index']]
            for n in range(1, len(current_df)):
                if int(current_df.iloc[n]['index']) == index[-1] +1:
                    index.append(current_df.iloc[n]['index'])
                else:
                    # apply imputation
                    self.impute_with_LSTM(index, model = model, feature = feature)
                    index = [current_df.iloc[n]['index']]
                
            self.impute_with_LSTM(index, model = model, feature = feature)