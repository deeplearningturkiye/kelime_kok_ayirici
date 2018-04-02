#!/usr/bin/python3

"""
==========================
Derin Öğrenme Tabanlı - seq2seq - Türkçe için kelime kökü bulma web uygulaması 
(tr_stemmer)
==========================

Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.
"""

# -*- coding: utf-8 -*- 
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from keras.models import model_from_json
import sys 
import os
import builtins
from urllib.request import urlopen
import utilities
import db

#initalize our flask app
app = Flask(__name__)
#app.config['JSON_AS_ASCII'] = False

#load model and weights
jstr = json.loads(open('kokbul.json').read())
model = model_from_json(jstr)
model.load_weights('kokbul-18-0.98.hdf5')

def kokBul(word):
	X = []

	w = utilities.encode(word)
	X.append(w)

	X = np.array(X)

	yp = model.predict(X)
	#print("Predicted : ",decode(yp[0]))
	return utilities.decode(yp[0])

	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route("/admin")
def admin():
	#records = db.getAllRecords()
	#return render_template('admin.html',records=records)
	return render_template('admin.html')


@app.route('/predict/',methods=['GET','POST'])
def predict():

	result = {"success": False}

	userText = request.get_data().decode('utf-8')
	print(userText)

	kok = kokBul(userText)

	#check first letter
	if(kok[:1] != utilities.lower(userText[:1])):
		kok = "bilmiyorum"

	#ger public ip
	user_ip = str(urlopen('http://ip.42.pl/raw').read()).replace("b'","").replace("'","")
	print("user_ip",user_ip)
	print("IP",request.remote_addr)
	
	#save database
	lastID = db.addRecord(userText, kok, user_ip)
	print("lastRecordID", lastID)

	result["predictions"] = {"kok": kok, "lastRecordID": lastID}
	result["success"] = True

	return jsonify(result)

#@app.route('/predictURL/',methods=['GET'])
#def predictURL():
#
#	word = request.args.get('q')
#	print(word)
#
#	kok = kokBul(word)
#
#	return jsonify(kok) #json.dumps(kok, ensure_ascii=False).encode('utf8') #


@app.route('/updateRecord/',methods=['POST'])
def updateRecord():
	_ID = request.form['_id']
	_IsTrue = request.form['_isTrue']
	_UserSuggestion = request.form['_userSuggestion']

	result = {"success": False}

	lastID = db.updateRecord(_ID, _IsTrue, _UserSuggestion)
	if lastID > 0:
		result["success"] = True

	return jsonify(result)

@app.route('/getAllRecords/',methods=['GET'])
def getAllRecords():

	records = db.getAllRecords()

	return jsonify({'data': records})

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)







