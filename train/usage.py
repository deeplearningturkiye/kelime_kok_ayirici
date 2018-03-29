# -*- coding: utf-8 -*-


"""

Module Name : usage kelime_kok_ayirici 
Date : Change
03/02/2017 : initial write
...
"""

import os
import numpy as np
import datetime
import random

import codecs
import numpy as np
import os
import time
import json
import pickle
from keras.models import model_from_json

jstr = json.loads(open('kokbul.json').read())
model = model_from_json(jstr)
model.load_weights('kokbul.hdf5')

fp = codecs.open('KOKBULTEST.txt','r',encoding='utf-8')
lines = fp.readlines()
lines  = [l.strip() for l in lines]

fp = open('datafile.pkl','rb')
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
	
kelimeler = []
kokler = []
for line in lines:
    words = line.split(',')
    kelimeler.append(words[0])
    kok = (words[1].split(':'))[0]
    kokler.append(kok)

words = []	
X = []
Y = []
ntest = 1000
for j in range(ntest):
	i = random.randint(0,len(kelimeler)-1)
	w = kelimeler[i]
	words.append(w)
	k = kokler[i]
	w = encode(w)
	#k = encode(k)
	X.append(w)
	Y.append(k)
X = np.array(X)
#Y = np.array(Y)

yp = model.predict(X)

ktrue = 0.0

for j in range(ntest):
	print "\n\nWord      : ",words[j]
	print "Kok       : ",Y[j]
	print "Predicted : ",decode(yp[j])
	if Y[j] == decode(yp[j]):
		ktrue +=1.0
		
print "Accuracy %0.02f"%(100*ktrue/ntest)
