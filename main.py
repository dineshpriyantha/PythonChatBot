import nltk
#nltk.download('all')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from flask import Flask, render_template, request
import pyodbc
 
app = Flask(__name__)

conn = pyodbc.connect('Driver={SQL Server};'
					  'Server=DESKTOP-NN1DR5L;'
					  'Database=chatbot;'
					  'Trusted_Connection=yes;')

global cursor
cursor = conn.cursor()
buffer = "SELECT * FROM chatbot.dbo.chat"
cursor.execute(buffer)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in cursor:
        for pattern in intent.patterns.split(","):
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent.tag)

        if intent.tag not in labels:
            labels.append(intent.tag)

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


## prediction

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat(userText):
    cursor.execute(buffer)

    while True:
        inp = userText#input(userText)
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag1 = labels[results_index]

        if results[results_index] > 0.7:         
            for tg in cursor:
                if tg.tag == tag1:
                    responses = tg.responses.split(",")

            return random.choice(responses)
        else:
            return "I didn't get that, try again."


@app.route("/")
def home():    
    return render_template("home.html")
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return chat(userText)

if __name__ == "__main__":
	
	HOST = '127.0.0.1'
	PORT = 4000      	#make sure this is an integer
 
	app.run(HOST, PORT)
