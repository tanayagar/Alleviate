# Alleviate

Health Services Platform using NLTK and Deep Neural Network using python.

# Description
A Chat Bot which uses Deep Neural Networks and NLTK to answer any health-related queries. Also uses Sentiment Analysis from Google Natural Language API and Places API to mine Hospital reviews and provide suggestions accordingly.

## Features
1. **Disease in Area Alert**: To find out the prevalent diseases and major outbreaks, RSS feed of TOI is scraped and summarised using TF-IDF. Geocoder is used to detect current location. ( RSS feed is available only for a few cities in India)
    > ```python diaa.py```
2. **Review Mining**: Using Google Places API, the reviews of all the places tagged "hospitals" in a particular geo-location are extracted. Sentiment Analysis is applied on recent hospital reviews. The user rating are ignored and only the reviews are considered to get the current response of people towards the place. The reviews and sentiment analysis data for 3 cities is stored in the _csv_ files.
    > ```python scrp.py```
3. **Query Diagnosis**: The user's query is chunked and passed to the Infermedica API to generate follow-up questions and disease prediction. After a set of questions is asked, a prognosis is given along with a list of top rated hospitals.
4. **Chat Bot** : Chatbot used for Alleviate is task-oriented and intension based. In chatbot framework,conversational intents are defined that would be useful for defining set of actions bot can manage.

## Demo
The Chatbot is started using ```python ChatBot.py```
The screenshots given below do not include the API output printed between queries

![Sending Query](https://i.imgur.com/RVRPvNa.png)

![First Question](https://i.imgur.com/falkmux.png)

![Second Question](https://i.imgur.com/wqkzmna.png)

![Third Question](https://i.imgur.com/egu0v9H.png)

![Prognosis](https://i.imgur.com/k6vdWFs.png)

![Follow up](https://i.imgur.com/zN9zbAt.png)

## Libraries/Frameworks/APIs Used:
- [Infermedica API](https://developer.infermedica.com/)
- [GCP Natural Language API](https://cloud.google.com/natural-language/)
- [NLTK](https://www.nltk.org/)

## Project Contributors
This project was done during my CSE4022 - Natural Language Processing by a team of three people
- [Tanay Agarwal](https://github.com/tanayagar/)
- [Vashisht Marhwal](https://github.com/vashishtmarhwal/)
- [Piyush Bamel](https://github.com/hydraplace)
