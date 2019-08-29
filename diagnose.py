from __future__ import print_function
import sys,os

import config
import json

config.setup_examples()
import infermedica_api


if __name__ == '__main__':
    sys.stdout = open(os.devnull,'w')
    api = infermedica_api.get_api()

    print('What is wrong?')
    RS=input()

    response = api.parse(RS)
    pjson=json.loads(str(response))

    idl=[]
    choice=[]
    sys.stdout = sys.__stdout__
    for data in pjson['mentions']:
        idl.append(data['id'])
        choice.append(data['choice_id'])


    request = infermedica_api.Diagnosis(sex='male', age=21)


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



    request = infermedica_api.Diagnosis(sex='male',age='21')

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



    request = infermedica_api.Diagnosis(sex='male',age='21')

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



    request = infermedica_api.Diagnosis(sex='male',age='21')

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

    