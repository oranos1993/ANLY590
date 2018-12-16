#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import statsmodels.api as sm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import random
from keras.callbacks import EarlyStopping


df = pd.read_csv('ver2.csv')

df1 = df[df['essay_set']==5]
df2 = pd.read_csv('training_set_rel3.tsv',sep = '\t',encoding = "ISO-8859-1")
df2 = df2[df2['essay_set']==5]
Y = df2.rater1_domain1 + df2.rater2_domain1
X = df1[['avg_sent_len','essay_length_no_stpwrd','num_synonym']]
df1['num_synonym_ratio'] = df1['num_synonym']/df1['essay_length']
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
model.summary()


Y = df2.rater1_domain1 + df2.rater2_domain1
X = df1[['avg_sent_len','num_synonym']]
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
model.summary()




# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(encoded_Y)
X = df1.drop(['essay_id', 'essay_set','essay','domain1_score','num_antonym'], axis=1)

#X = df1[['num_sent','avg_sent_len','essay_length','essay_length_no_stpwrd','len_wrd',
#         'len_wrd_stpwrd','len_uniwrd','num_uniwrd','num_uniwrd_stpwrd','num_synonym']]
X1 = X.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)
Y = (df2.rater1_domain1 + df2.rater2_domain1).values.astype(int)
X1 = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

rand_index = random.sample(range(len(Y)), len(Y))
X_shuffle = X1[rand_index]
Y_shuffle = Y[rand_index]
split_index = int(len(Y)*0.8)
X_train = X_shuffle[:split_index,:]
Y_train = Y_shuffle[:split_index,]
X_test = X_shuffle[split_index:,:]
Y_test = Y_shuffle[split_index:,]

# LSTM
model = Sequential()
#model.add(LSTM(64, input_shape=(1, 10)))
model.add(LSTM(64, input_shape=(1, 71)))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=RMSprop())
model.fit(X_train, Y_train, epochs=50, batch_size=8, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.000001)],
          validation_data=[X_test, Y_test])
y_pred = model.predict(X_test)
y_pred = [round(float(i)) for i in y_pred]
sum(1 for x,y in zip(y_pred,Y_test) if (x >= y-1) and (x <= y+1)) / len(Y_test)


X = df1.drop(['essay_id', 'essay_set','essay','domain1_score','num_antonym'], axis=1)
#X = df1[['num_sent','avg_sent_len','essay_length','essay_length_no_stpwrd','len_wrd',
#         'len_wrd_stpwrd','len_uniwrd','num_uniwrd','num_uniwrd_stpwrd','num_synonym']]
X1 = X.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)
Y = (df2.rater1_domain1 + df2.rater2_domain1).values.astype(int)

rand_index = random.sample(range(len(Y)), len(Y))
X_shuffle = X1[rand_index]
Y_shuffle = Y[rand_index]
split_index = int(len(Y)*0.8)
X_train = X_shuffle[:split_index,:]
Y_train = Y_shuffle[:split_index,]
X_test = X_shuffle[split_index:,:]
Y_test = Y_shuffle[split_index:,]


# Deep ANN
model = Sequential()
#model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, input_dim=71, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer=RMSprop())
model.fit(X_train, Y_train, epochs=100, batch_size=8, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.000001)],
          validation_data=[X_test, Y_test])
y_pred = model.predict(X_test)
y_pred = [round(float(i)) for i in y_pred]
sum(1 for x,y in zip(y_pred,Y_test) if (x >= y-1) and (x <= y+1)) / len(Y_test)


# LSTM on Classification 
Y = np_utils.to_categorical(encoded_Y)
X = df1.drop(['essay_id', 'essay_set','essay','domain1_score','num_antonym'], axis=1)

#X = df1[['num_sent','avg_sent_len','essay_length','essay_length_no_stpwrd','len_wrd',
#         'len_wrd_stpwrd','len_uniwrd','num_uniwrd','num_uniwrd_stpwrd','num_synonym']]
X1 = X.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)
X1 = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
rand_index = random.sample(range(len(Y)), len(Y))
X_shuffle = X1[rand_index]
Y_shuffle = Y[rand_index]
split_index = int(len(Y)*0.8)
X_train = X_shuffle[:split_index,:]
Y_train = Y_shuffle[:split_index,]
X_test = X_shuffle[split_index:,:]
Y_test = Y_shuffle[split_index:,]

model = Sequential()
#model.add(LSTM(64, input_shape=(1, 10)))
model.add(LSTM(64, input_shape=(1, 71)))
model.add(Dense(128,activation='relu'))
model.add(Dense(9,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)],
          validation_data=[X_test, Y_test])
y_pred = model.predict(X_train)


