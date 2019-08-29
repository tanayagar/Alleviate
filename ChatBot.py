from __future__ import print_function
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd

def setup_examples():
    """
    Setup environment to easily run examples.
    API credentials needs to be provided here in order
    to set up api object correctly.
    """
    try:
        import infermedica_api
    except ImportError:
        import sys
        import os

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import infermedica_api

    # !!! ENTER YOUR CREDENTIALS HERE !!!
    infermedica_api.configure({
        'app_id': 'b2a6613b',
        'app_key': '047a70b9e8efd5691f3969301ecb3a18',
        'dev_mode': True  # Use only during development on production remove this parameter
    })

    import logging

    # enable logging of requests and responses
    try:
        import httplib
        httplib.HTTPConnection.debuglevel = 1
    except ImportError:
        import http.client
        http.client.HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def diag(sex,age):
    import sys,os
    import json

    setup_examples()
    import infermedica_api
    if(True):
        #sys.stdout = open(os.devnull,'w')
        api = infermedica_api.get_api()

        print('What is wrong?')
        RS=input()

        response = api.parse(RS)
        pjson=json.loads(str(response))

        idl=[]
        choice=[]
        #sys.stdout = sys.__stdout__
        for data in pjson['mentions']:
            idl.append(data['id'])
            choice.append(data['choice_id'])


        request = infermedica_api.Diagnosis(sex, age)


        for i in range(len(idl)):
            request.add_symptom(idl[i],choice[i])


        request = api.diagnosis(request)
        pjson=json.loads(str(request))

        response=pjson

        names=""
        furt=""
        for data in response:
            if(data == 'conditions'):
                names=response[data][0]['common_name']
            if (data == 'question'):
                furt=response[data]['text']




        print(furt+"\n")
        query=input()
        query=query+" "+names.split(',')[0]
        RS=RS+", "+query





        response = api.parse(RS)
        pjson=json.loads(str(response))

        idl=[]
        choice=[]
        sys.stdout = sys.__stdout__
        for data in pjson['mentions']:
            idl.append(data['id'])
            choice.append(data['choice_id'])



        request = infermedica_api.Diagnosis(sex,age)

        for i in range(len(idl)):
            request.add_symptom(idl[i],choice[i])


        request = api.diagnosis(request)
        pjson=json.loads(str(request))

        response=pjson

        names=""
        furt=""
        for data in response:
            if(data == 'conditions'):
                names=response[data][0]['common_name']
            if (data == 'question'):
                furt=response[data]['text']


        print(furt+"\n")
        query=input()
        query=query+" "+names.split(',')[0]
        RS=RS+", "+query





        response = api.parse(RS)
        pjson=json.loads(str(response))

        idl=[]
        choice=[]
        sys.stdout = sys.__stdout__
        for data in pjson['mentions']:
            idl.append(data['id'])
            choice.append(data['choice_id'])



        request = infermedica_api.Diagnosis(sex,age)

        for i in range(len(idl)):
            request.add_symptom(idl[i],choice[i])


        request = api.diagnosis(request)
        pjson=json.loads(str(request))

        response=pjson

        names=""
        furt=""
        for data in response:
            if(data == 'conditions'):
                names=response[data][0]['common_name']
            if (data == 'question'):
                furt=response[data]['text']


        print(furt+"\n")
        query=input()
        query=query+" "+names.split(',')[0]
        RS=RS+", "+query





        response = api.parse(RS)
        pjson=json.loads(str(response))

        idl=[]
        choice=[]
        sys.stdout = sys.__stdout__
        for data in pjson['mentions']:
            idl.append(data['id'])
            choice.append(data['choice_id'])



        request = infermedica_api.Diagnosis(sex,age)

        for i in range(len(idl)):
            request.add_symptom(idl[i],choice[i])


        request = api.diagnosis(request)
        pjson=json.loads(str(request))

        response=pjson

        names=""
        furt=""
        for data in response:
            if(data == 'conditions'):
                print(response[data][0]['common_name'])
            if (data == 'question'):
                print(response[data]['text'])

import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

string = "My name is Piyush and I lives in Delhi. My phone no. 9992014705. My email is piyushbamel1@gmail.com"

def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)
def extract_name_place(string):
    names=[]
    place=[]
    ne_tree = ne_chunk(pos_tag(word_tokenize(string)))
    for sent in nltk.sent_tokenize(string):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                if(chunk.label()=="PERSON"):
                    for c in chunk:
                        names.append(c[0])
                if(chunk.label()=="GPE"):
                    for c in chunk:
                        place.append(c[0])
    return names,place
numbers = extract_phone_numbers(string)
emails = extract_email_addresses(string)
names,place = extract_name_place(string)
print(numbers,emails,names,place)


import json
with open('intent.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?','.']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

#reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intent.json') as json_data:
    intents = json.load(json_data)


# load our saved model
model.load('./model.tflearn')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.8
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return print(random.choice(i['responses']))
            return results.pop(0)

flag=0
while(True):
    if(flag==0):
        user_input = input("Waiting for your input............")
        flag=1
    else:
        user_input = input(resp)    
    group=classify(user_input)
    if not group[0][0]:
        resp="I did not understand you. Please be more clear"
    elif group[0][0]=="info":
        numbers = extract_phone_numbers(user_input)
        emails = extract_email_addresses(user_input)
        names,place = extract_name_place(user_input)
        print(numbers,emails,names,place)
        resp="Thanks for you input"
    elif group[0][0]=="help":
        f=0
        while(f==0):
            gender=str(input("What is your gender? M or F"))
            gender=gender.split(' ')
            g=[e for e in ["M","m","male","Male","F","f","female","Female"] if e in gender]
            if(len(g)>0):
                s=[e for e in ["M","m","male","Male","F","f","female","Female"] if e in gender]
                gender=s[0]
                if(gender =="M" or gender=="m" or gender=="Male"):
                    gender="male"
                else:
                    gender="female"
                f=1
            else:
                print("Type correctly")
        f=0
        while(f==0):
            age=str(input("What is your age?"))
            age=[int(s) for s in age.split() if s.isdigit()]
            if(age[0]>0 and age[0]<100):
                f=1
                age=age[0]
            else:
                print("Type correctly")
        print(gender,age)
        diag(gender,age)
        resp="Do you want to suggest you are doctor"
    elif group[0][0]=="goodbye":
        resp=response(user_input)
        print(response)
        break
    elif group[0][0]=="yes":
        user_input=input("Tell me where you live")
        name,place=extract_name_place(user_input)   
        df=pd.read_csv('reviews.csv')
        df=df[df.loc[:,'1']==place[0]]
        print(df.head(3))
        print("Take Care")
        break
    else:
        resp=response(user_input)
