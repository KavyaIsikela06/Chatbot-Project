import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
import os
# Load intents file
with open("intents.json") as file:
    data=json.load(file)
# Load preprocessed data if available
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output=pickle.load(f)
except:
    # Initialize lists
    words=[];labels=[];docs_x=[];docs_y=[]
    # Tokenize patterns and collect tags
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    # Stem and clean words
    words=[stemmer.stem(w.lower()) for w in words if w!="?"]
    words=sorted(list(set(words)))
    labels=sorted(labels)
    training=[];output=[]
    out_empty=[0 for _ in range(len(labels))]
    # Create bag-of-words and one-hot outputs
    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)
        output_row=out_empty[:]
        output_row[labels.index(docs_y[x])]=1
        training.append(bag)
        output.append(output_row)
    training=np.array(training);output=np.array(output)
    # Save preprocessed data
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)
# Reset graph (for TensorFlow 1.x compatibility)
tensorflow.compat.v1.reset_default_graph()
# Build model: input → hidden layers → output with softmax
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)
model=tflearn.DNN(net)
# Train or load existing model
if os.path.exists("model.tflearn.index"):
    model.load("model.tflearn")
else:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")
# Convert input text to bag-of-words
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
    return np.array(bag)
# Chat function (interactive loop)
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp=input("You: ")
        if inp.lower()=="quit":
            break
        # Predict intent
        results=model.predict([bag_of_words(inp,words)])
        results_index=np.argmax(results)
        tag=labels[results_index]
        # Match response
        for tg in data["intents"]:
            if tg['tag']==tag:
                responses=tg['responses']
        print(random.choice(responses))
chat()
