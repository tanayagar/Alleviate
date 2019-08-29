import requests,json
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from langdetect import detect
import nltk
import re
import pandas as pd

df = pd.DataFrame()
df2 = pd.DataFrame()
def cleaner(text):
	new_text=""
	# words=set(nltk.corpus.words.words())
	# text=" ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
	# return text
	try:
		if(detect(text)=='en'): ##Check Locale
			for s in text:
				if(s.isalnum() or s in [' ','?','.','#','$','(',')']):
					new_text+=s
		else:
			return ""
		return new_text
	except:
		return ""

bist=[]

def print_result(annotations):
	##print("Phase \n",annotations)
	for i in annotations.sentences:
		bist.append(i.sentiment.score) ##Extact the sentiment score of each sentence
	return sum(bist) ##Total the sentment score
	return 0


city='Delhi' ## Will be updated as input from the chatbot
url = "https://maps.googleapis.com/maps/api/place/textsearch/json?input=Hospital%20"+city+"&inputtype=textquery&fields=formatted_address,name,rating,placeid&key=XXXXXXXX" ##API key to fetch hospitals using the google places API

r=requests.get(url)
x=r.json()

y=x['results'] ## Storing the results child in the json

for i in range(3):
	#print(y[i]['name'],y[i]['place_id'])
	url2="https://maps.googleapis.com/maps/api/place/details/json?key=XXXXXXXXXX&language=en&placeid="+y[i]['place_id']
	## Using the place API to get details of the reviews of the hospital in the given city
	r=requests.get(url2)

	x=r.json()
	z=x['result']['reviews'] ## Extracting only the reviews from the details
	
	client = language.LanguageServiceClient() ##Google AutoML Nalutral Language Client Initialization
	r1=0 ## Variable to store the total sentiment value for the hospital
	for p in z:  
		## Clean the text to remove non english characters
		content=cleaner(p['text'])
		#print(content+"  ")
		document = types.Document(content=content,type=enums.Document.Type.PLAIN_TEXT)
		try:
			annotations = client.analyze_sentiment(document=document) ## Use google API to perfrom sentiment analysis
			val=print_result(annotations) 
			r1=r1+val ## Append to the total sentiment value of the hospital
			l=[[y[i]['name'],city,content,val]] ## Create a list
			df=df.append(l) ## Append details to the dataframe
		except: ##Exception to prevent non english error 
			r1=r1+0
		bist=[]
		#print("\n\n")
	#print("For Hospital ",y[i]['name'],r1/len(z),'\n')
	r1=r1/len(z) ## Calculate the average sentiment value
	l=[[y[i]['name'],city,r1]] 
	df2=df2.append(l) ##Create dataset of the hospital with the average sentiment values

df.to_csv("reviews.csv",mode='a') ## Append to database
df2.to_csv("sentiment.csv",mode='a') ## Append to database
