import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import csv
import warnings
from flask import Flask, jsonify, request
from flask_cors import CORS
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print ("cross result : ")
scores = cross_val_score(clf, x_test, y_test, cv=3)
#print (scores)
print (scores.mean())

y_pred = clf.predict(x_test)

print("Confusion Matrix is : ")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("Classification Report : ")
print(classification_report(y_test, y_pred))


model=SVC()
model.fit(x_train,y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        return {"severity" : "high"}
    else:
        return {"severity" : "low"}


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item


def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    print("Decision Tree X train : ", X_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1


    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names, disease_input, num_days):

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            disease_input=cnf_dis[0]
            break
        else:
            print("Enter valid symptom.")
    
    def recurse(node, depth):
        global syms_given, present_disease
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
        
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            syms_given =list(symptoms_given)

            confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            print("confidence level is " + str(confidence_level))

    recurse(0, 1)

    return {'possibleSymptoms' : syms_given, 'presentDisease': present_disease[0]}

getSeverityDict()
getDescription()
getprecautionDict()

app = Flask(__name__)
CORS(app)

@app.route('/findPossibleSymptoms', methods=['POST'])
def getSymptoms():
    global clf, cols
    ndays = request.json['num_days']
    dinput = request.json['disease_input']
    return tree_to_code(clf, cols ,dinput, ndays)

@app.route('/find', methods=['POST'])
def getSecond():
    symptomsHave = request.json['have_symptoms']
    numDays = int(request.json['ndays'])
    return calc_condition(symptomsHave, numDays)

@app.route('/getDescription')
def getDesc():
    global present_disease, description_list
    return {'diseaseDesc' : description_list[present_disease[0]], 'disease':present_disease[0], 'precautions' : precautionDictionary[present_disease[0]]}


if __name__ == "__main__":
    app.run(debug= True)