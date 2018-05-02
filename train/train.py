# -*- coding: utf-8 -*-


"""
Module Name : kelime_kok_ayirici training 
Date : Change
03/02/2017 : initial write
...
"""

import numpy as np
import os
import time
import json
import pickle
import pandas as pd

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import LSTM, BatchNormalization,concatenate,BatchNormalization,multiply
from keras.layers import TimeDistributed, Bidirectional,Reshape
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint,CSVLogger
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/kelime_kok.csv',encoding='utf-8')
kelimeler = df.kelime.tolist()
kokler = [ x.split(':')[0] for x in df.kok.tolist()]
del df


fp = open('../data/datafile.pkl','rb')
data = pickle.load(fp)
fp.close()

chars = data['chars']
charlen = data['charlen']
maxlen = data['maxlen']
charlen = len(chars)
maxlen = 22

def encode(word,maxlen=22,is_pad_pre=False):
    wlen = len(word)
    if wlen > maxlen:
        word = word[:maxlen]
        
    word = word.lower()
    pad = maxlen - len(word)
    if is_pad_pre :
        word = pad*' '+word   
    else:
        word = word + pad*' '
    mat = []
    for w in word:
        vec = np.zeros((charlen))
        if w in chars:
            ix = chars.index(w)
            vec[ix] = 1
        mat.append(vec)
    return np.array(mat)    

def decode(mat):
    word = ""
    for i in range(mat.shape[0]):
        word += chars[np.argmax(mat[i,:])]
    return word.strip()
		
X = []
Y = []
for w,k in zip(kelimeler,kokler):
    w = encode(w,maxlen)
    k = encode(k,maxlen)
    X.append(w)
    Y.append(k)
X = np.array(X)
Y = np.array(Y)

# Split training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2017)

# model description
harf_input = Input(shape=(maxlen,charlen))
flat = Flatten()(harf_input)
atn = Dense(maxlen*charlen)(flat)
atn = Reshape((maxlen,charlen))(atn)
attention_mul = multiply([harf_input,atn])
lstm_harf_out = LSTM(64,return_sequences=True)(attention_mul)                   
time_out = TimeDistributed(Dense(charlen))(lstm_harf_out)
time_out = Activation('softmax')(time_out)
                   
model = Model(inputs=harf_input, outputs=time_out)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

mjs = model.to_json()
with open('models/kokbul.json', 'w') as outfile:
    json.dump(mjs, outfile)
	
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=9, verbose=1, mode='auto')
redc = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1, mode='auto', min_lr=0.00001)

model_checkpoint = ModelCheckpoint(filepath="models/kokbul-{epoch:02d}-{val_acc:.3f}.hdf5", 
                                   monitor='val_acc', save_best_only=True,mode='auto',save_weights_only=True,verbose=1)

logfile = 'models/new_traing_'+('_'.join([str(x) for x in time.localtime()[:6]]))+'.log'
csv_logger = CSVLogger(logfile)
cblist = [model_checkpoint,redc,csv_logger]

model.fit(np.float32(x_train), np.float32(y_train),
          batch_size=16,
          epochs=33,
          validation_data=(np.float32(x_test), np.float32(y_test)),callbacks=cblist,
          shuffle=True)