# -*- coding: utf-8 -*-


"""

Module Name : usage kelime_kok_ayirici 
Date : Change
03/02/2017 : initial write
02/05/2017 : new data file and trained models updated
...
"""

import os
import numpy as np
import datetime
import random

import pandas as pd
import numpy as np
import os
import time
import json
import pickle
from keras.models import model_from_json

jstr = json.loads(open('models/kokbul.json').read())
model = model_from_json(jstr)
model.load_weights('models/kokbul-16-0.983.hdf5')

df = pd.read_csv('../data/kelime_kok.csv',encoding='utf-8')
kelimeler = df.kelime.tolist()
kokler = [ x.split(':')[0] for x in df.kok.tolist()]

fp = open('../data/datafile.pkl','rb')
data = pickle.load(fp)
fp.close()

chars = data['chars']
charlen = data['charlen']
maxlen = data['maxlen']

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
	
words = []	
X = []
Y = []
ntest = 1000
for j in range(ntest):
	i = random.randint(0,len(kelimeler)-1)
	w = kelimeler[i]
	words.append(w)
	k = kokler[i]
	w = encode(w,maxlen)
	X.append(w)
	Y.append(k)
X = np.array(X)

yp = model.predict(X)

ktrue = 0.0

for j in range(ntest):
	print "\n\nWord      : ",words[j]
	print "Kok       : ",Y[j]
	predicted_kok = decode(yp[j])
	
	# kural tabanlı kontrol ve düzeltme
	if len(words[j]) < len(predicted_kok) :
		predicted_kok = predicted_kok[:len(words[j])]
		
	print "Predicted : ", predicted_kok
	if Y[j] == predicted_kok:
		ktrue +=1.0
		
print "Accuracy %0.02f"%(100*ktrue/ntest)
