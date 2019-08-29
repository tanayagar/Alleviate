import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import geocoder
import re
import heapq
myloc = geocoder.ip('me')
loc = myloc.latlng
min = []
#getting the location of the device
longlat = {'Mumbai': [19.075983, 72.877655], 'Delhi': [28.704060, 77.102493], 'Bangalore':[12.971599, 77.594566], 'Hyderabad':[17.385044, 78.486671], 'Chennai':[13.082680, 80.270721], 'Ahemdabad':[23.022505, 72.571365]}
# assigning longitude and latitude for cities

mint=[]
summaryList = []
for i in longlat:
        min.append(np.array(longlat[i])-np.array(loc))
for i in range(len(min)):
        mint.append(round(min[i][0]+min[i][1]))
#Extracting your location from the mininum distance between longitude and latitude
city = list(dict.keys(longlat))[mint.index(0)]
rssfeed = {'Mumbai': -2128838597, 'Delhi': -2128839596, 'Bangalore': -2128833038, 'Hyderabad': -2128816011, 'Chennai': 2950623, 'Ahemdabad': -2128821153}
#storing the rss number for the cities from times of india rss feeder
import feedparser
from nltk.corpus import wordnet 
synonyms = [] 
#using nltk to form a word set for synonyms for word outbreak and spread
#using wordnet inside nltk library to get the required words
for syn in wordnet.synsets("spread"):
    for l in syn.lemmas():
        synonyms.append(l.name())  
  
spread = set(synonyms) 
import nltk
from nltk.corpus import wordnet 
synonyms = [] 
for syn in wordnet.synsets("outbreak"): 
    for l in syn.lemmas(): 
        synonyms.append(l.name())  
#taking a union of the word synonym set formed
out = set(synonyms)
out = out.union(spread)
probable = []
d = feedparser.parse('https://timesofindia.indiatimes.com/rssfeeds/'+str(rssfeed[city])+'.cms')#parsing the required titles from a cerating location, that contains the news regarding the disease outbreak


for post in d.entries:
    for i in out:
        if(i in post.title):
                soup = BeautifulSoup(post.summary, 'html.parser')
                probable.append(soup.get_text)
stopwords = nltk.corpus.stopwords.words('english')
#Summarising the News title summary 
for i in range(len(probable)):
        if(probable[i]):
                probable[i] = re.sub(r'\[[0-9]*\]', ' ', probable[i])
                probable[i] = re.sub(r'\s+', ' ', probable[i])
                sentence_list = nltk.sent_tokenize(probable[i])
                word_frequencies = {}  
                for word in nltk.word_tokenize(probable[i]):  
                        if word not in stopwords:
                                if word not in word_frequencies.keys():
                                        word_frequencies[word] = 1
                                else:
                                        word_frequencies[word] += 1
                sentence_scores = {}
                for sent in sentence_list:
                        for word in nltk.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                        if len(sent.split(' ')) < 30:
                                                if sent not in sentence_scores.keys():
                                                        sentence_scores[sent] = word_frequencies[word]
                                                else:
                                                        sentence_scores[sent] += word_frequencies[word] 
                summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
                summary = ' '.join(summary_sentences)  
                summaryList.append(summary)


        
if(summaryList):
        print(probable, summaryList)
else:
        print("No current outbreaks ")

cd = ['HIV/AIDS', 'Chickenpox', 'Chronic Fatigue Syndrome', 'Common Cold', 'Diphtheria', 'E. coli', 'Giardiasis', 'Infectious Mononucleosis', 'Flu', 'Lyme Disease', 'Malaria', 'Measles', 'Meningitis','Mumps', 'Polio', 'Pneumonia','Rocky Mountain Spotted Fever', 'Salmonella Infections','Severe Acute Respiratory Syndrome','Sexually Transmitted Diseases','Shingles', 'Zoster', 'Tetanus', 'Toxic Shock Syndrome', 'Tuberculosis', 'Viral Hepatitis', 'West Nile Virus','Whooping Cough']
prob = [x.split(" ") for x in probable]

cd = [x.lower() for x in cd]
for x in range(len(prob)):
    if(set(prob[x]).intersection(cd)):
        print(str(set(prob[x]).intersection(cd))[2:5]+" Alert")