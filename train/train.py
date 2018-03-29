# -*- coding: utf-8 -*-


"""
Module Name : kelime_kok_ayirici training 
Date : Change
03/02/2017 : initial write
...
"""

import codecs
import numpy as np
import os
import time
import json
import pickle

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import LSTM, BatchNormalization,concatenate,BatchNormalization,multiply
from keras.layers import TimeDistributed, Bidirectional,Reshape
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint,CSVLogger
from sklearn.model_selection import train_test_split

fp = codecs.open('KOKBULTEST.txt','r',encoding='utf-8')
lines = fp.readlines()
lines  = [l.strip() for l in lines]

chars = list(set(''.join(lines)))
chars.append(' ')
chars.append('q')
chars.sort()
chars =''.join(chars).strip()
chars = chars.replace(',','')
chars = ' '+chars

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
	
kelimeler = []
kokler = []
for line in lines:
    words = line.split(',')
    kelimeler.append(words[0])
    kok = (words[1].split(':'))[0]
    kokler.append(kok)
	
X = []
Y = []
for w,k in zip(kelimeler,kokler):
    w = encode(w)
    k = encode(k)
    X.append(w)
    Y.append(k)
X = np.array(X)
Y = np.array(Y)

# Split training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2017)

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

data = {}
data['maxlen'] = maxlen
data['charlen'] = charlen
data['chars'] = chars
fp = open('datafile.pkl','wb')
pickle.dump(data,fp)
fp.close()

mjs = model.to_json()
with open('kokbul.json', 'w') as outfile:
    json.dump(mjs, outfile)
	
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=9, verbose=1, mode='auto')
redc = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1, mode='auto', min_lr=0.00001)

model_checkpoint = ModelCheckpoint(filepath="kokbul-{epoch:02d}-{val_acc:.2f}.hdf5", 
                                   monitor='val_acc', save_best_only=True,mode='auto',save_weights_only=True,verbose=1)

logfile = 'new_traing_'+('_'.join([str(x) for x in time.localtime()[:6]]))+'.log'
csv_logger = CSVLogger(logfile)
cblist = [model_checkpoint,redc,csv_logger]

model.fit(np.float32(x_train), np.float32(y_train),
          batch_size=1,
          epochs=33,
          validation_data=(np.float32(x_test), np.float32(y_test)),callbacks=cblist,
          shuffle=True)