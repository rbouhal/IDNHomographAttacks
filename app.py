from re import X
from tkinter import Y
from flask import Flask, render_template, request, jsonify
import string
import csv
import os
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import math
import pandas as pd
import numpy as np

app = Flask(__name__)

# Renders the Home page
@app.route('/')
def home():
    return render_template('index.html')

# Renders the About Us page
@app.route('/about')
def about():
    return render_template('about.html')

# Renders the How It Works page
@app.route('/howitworks')
def how_it_works():
    return render_template('howitworks.html')

# Retrieves the domain from the user input
@app.route('/check_domain', methods=['POST'])
def check_domain():
    domain = request.form['domain']
    print(domain)
    domain_data, validity, accuracy = process_domain(domain)
    return render_template('result.html', domain=domain, validity=validity, accuracy=accuracy*100, details=domain_data)

# Processess the domain for feature extraction and prediction
def process_domain(domain):
    domain_data = calculate_domain_characteristics(domain)
    # Create a copy of domain_data for prediction purposes
    prediction_domain_data = {key: value for key, value in domain_data.items()}
    validity, accuracy = predict_domain_validity(domain, prediction_domain_data)
    print(accuracy)
    print(validity)
    domain_data['domain_label'] = 1 if validity == 'valid' else 0
    write_to_csv(domain_data, validity, accuracy)

    #Run the ID3 algorithm on the domain and display the new entropy
    pred, upEntropy = predict_ID3(domain)
    print(upEntropy)
    
    return domain_data, validity, accuracy

# Calculates the domain features
def calculate_domain_characteristics(domain):
    characteristics = {
        'domain': domain,
        'domain_length': len(domain),
        'domain_hash': hash_domain(domain.split('.')[0]),
        'domain_char_count': sum(c.isalpha() for c in domain.split('.')[0]),
        'domain_digit_count': sum(c.isdigit() for c in domain.split('.')[0]),
        'non_ascii_char_count': count_non_ascii_chars(domain.split('.')[0]),
        'domain_tld': domain.split('.')[-1] if '.' in domain else ''
    }
    return characteristics

# Helper to determine if domain has non-ascii characters before TLD
def count_non_ascii_chars(s):
    return sum(1 for c in s if ord(c) > 127)

# Helper to determine the integer hash of the part of the domain before the TLD.
def hash_domain(domain):
    hash_full = hashlib.sha256(domain.encode('utf-8')).hexdigest()
    truncated_hash = hash_full[:16]  # Truncate to first 16 hex characters, which is 64 bits
    return int(truncated_hash, 16)

# Write's the domain + calculated characteristics to the respective csv file
def write_to_csv(domain_data, validity, accuracy):
    if accuracy >= .75:
        filename = 'static/valid-domains.csv' if validity == 'valid' else 'static/invalid-domains.csv'
        fieldnames = list(domain_data.keys())
        write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(domain_data)

#train model
def train_model(model):
    #concatenate the two dataframes
    df1 = pd.read_csv('static/valid-domains.csv')
    df2 = pd.read_csv('static/invalid-domains.csv')
    df = pd.concat([df1, df2])

    #prepare the data
    X = df[['domain_length','domain_hash','domain_char_count','domain_digit_count','non_ascii_char_count']]
    y = df['domain_label']

    #split the dataset into training set and test set
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    #create a DecisionTreeClassifier object
    model = DecisionTreeClassifier()

    #train the model using the training sets
    model.fit(XTrain, yTrain)
    return model, X, y
   

# Implement ML modle here given the domain, based on prediction be sure to return 'valid' or 'invalid'
def predict_domain_validity(domain, domain_data):
    model = None
    trained_model, X, y = train_model(model)
    
    # ensure the model is trained
    if trained_model is None:
        return 'Model not trained'

    #remove 'domain' and 'domain_tld' from the dictionary as not needed in model
    domain_data.pop('domain', None)
    domain_data.pop('domain_tld', None)

    #convert to DataFrame
    domain_df = pd.DataFrame([domain_data])

    #predict the validity
    prediction = trained_model.predict(domain_df)
    
    #*get accuracy using cross validation*
    #10 folds
    scores = cross_val_score(trained_model, X, y, cv=10)
    accuracy = scores.mean()

    #return 'valid' if prediction is 1, 'invalid' otherwise
    return 'valid' if prediction[0] == 1 else 'invalid', accuracy

#this function trains the ID3 decision tree with the current csv data
def trainID3():
    df1 = pd.read_csv('static/valid-domains.csv')
    df2 = pd.read_csv('static/invalid-domains.csv')
    #format the data
    valNP = df1.to_numpy()
    invNP = df2.to_numpy()
    valNP = np.delete(valNP, 0, axis=1)
    valNP = np.delete(valNP, -1, axis=1)
    invNP = np.delete(invNP, 0, axis=1)
    invNP = np.delete(invNP, -1, axis=1)
    val = dataAnalysis(valNP)
    inv = dataAnalysis(invNP)
    valCount = 0
    for ind in range(len(val)):
        valCount = valCount + 1
        val[ind][33] = 1
    fullData = np.concatenate((val, inv), axis=0)

    ctLst = []
    for ele in fullData[0]:
        ctLst.append(2)
    totalCount = 0
    for ele in fullData:
        totalCount = totalCount + 1
    tree = []
    level = 0
    #train the model using formated data
    return ID3(fullData, ctLst, valCount, totalCount - valCount, tree, level), valCount, totalCount - valCount
#predicts a domain using the ID3 algorithm given a csv data array
#if the funciton returns 1, domain is valid, if it returns 0 it is invalid
#also returns a previously calculated accuracy
def predict_ID3(domain):
  tree, class1, class0 = trainID3()
  csvInput =  calculate_domain_characteristics(domain)
  input = np.array(list(csvInput))
  conIn = inputConversion(input)
  classificaiton = ID3Classify(conIn, tree)
  if classificaiton == 0:
    currEntropy = entropy(class0 + 1, class1)
  else:
    currEntropy = entropy(class0, class1 + 1)
  return classificaiton, currEntropy


#this function converts some of the data features to a binary format
def dataAnalysis(data):
  numData = np.delete(data, -1, axis=1)
  binData = np.zeros((len(numData), 34))
  for ind, ele in enumerate(numData):
    #feature 1 extrapolation
    fin = 0
    for i in range(15):
      if i == ele[0]:
        binData[ind, i] = 1
        fin = 1
    if fin == 0:
      binData[ind, 15] = 1
    #feature 2 extrapolation
    fin = 0
    for i in range(8):
      if i == ele[1]:
        binData[ind, 16 + i] = 1
        fin = 1
    if fin == 0:
      binData[ind, 24] = 1
    #feature 3 extrapolation
    fin = 0
    for i in range(4):
      if i == ele[2]:
        binData[ind, 25 + i] = 1
        fin = 1
    if fin == 0:
      binData[ind][29] = 1
    #feature 6 extrapolation
    fin = 0
    for i in range(2):
      if i == ele[5]:
        binData[ind, 30 + i] = 1
        fin = 1
    if fin == 0:
      binData[ind, 32] = 1
  return binData

#this function implements the ID3 algorithm for a decision tree
def ID3(data, catNumOptionsLst, c1, c0, tree, level):
  infoGainLst = []
  classLabel = np.array(data[:, -1])
  if c1 == 0:
    tree.append([-1, level])
    return
  if c0 == 0:
    tree.append([-2, level])
    return
  if len(catNumOptionsLst) == 0:
    c0Count = 0
    c1Count = 0
    for ele in data:
      if ele[-1] == 0:
        c0Count = c0Count + 1
      else:
        c1Count = c1Count + 1
    if c0Count < c1Count:
      tree.append([-2, level])
    else:
      tree.append([-1, level])
    return
  for ind,op in enumerate(catNumOptionsLst):
    currCat = np.array(data[:, ind])
    catLabel = np.column_stack((currCat, classLabel)) #combine the category data and label data
    sortInd = np.argsort(catLabel[:,0]) #sort categores for spliting
    sCat = catLabel[sortInd]
    currInfoG = 0
    overallEntropy = entropy(c0, c1)
    #now calculate information gain of current feature
    for i in range(op):
      mask = (sCat[:, 0] == i)
      catI = sCat[mask]
      if catI.size != 0:
        splitClass1 = 0
        if(np.size(catI) == 2):
          splitClass1 = catI[0][1]
        else:
          splitClass1 = catI.sum(axis=1)[1]
        currInfoG += np.size(catI)/np.size(sCat) * entropy(splitClass1, np.size(catI) - splitClass1)
      else:
        currInfoG = 999999
        break
    currInfoG = overallEntropy - currInfoG
    infoGainLst.append([currInfoG, ind])
  #determine the highest information gain
  maxV = max([sublist[0] for sublist in infoGainLst])
  if maxV == 0:
    print("Limit reached")
    c0Count = 0
    c1Count = 0
    for ele in data:
      if ele[-1] == 0:
        c0Count = c0Count + 1
      else:
        c1Count = c1Count + 1
    if c0Count < c1Count:
      tree.append([-2, level])
    else:
      tree.append([-1, level])
    return
  maxInd = -1
  for ele in infoGainLst:
    if maxV in ele:
      maxInd = ele[1]
  currCat = np.array(data)

  sortInd = np.argsort(currCat[:,maxInd]) #sort categores for spliting
  sCat = currCat[sortInd]
  zeroSplt = []
  oneSplt = []
  #now split the two max info lists
  for ln in sCat:
    if ln[maxInd] == 1:
      oneSplt.append(ln.tolist())
    else:
      zeroSplt.append(ln.tolist())
  sp0 = np.array(zeroSplt)
  sp1 = np.array(oneSplt)
  tree.append([maxInd, level])
  sp0C0 = 0
  sp0C1 = 0
  sp1C0 = 0
  sp1C1 = 0
  for ele in sp0[:, -1]:
    if ele == 1:
      sp0C1 = sp0C1 + 1
    else:
      sp0C0 = sp0C0 + 1
  for ele in sp1[:, -1]:
    if ele == 1:
      sp1C1 = sp1C1 + 1
    else:
      sp1C0 = sp1C0 + 1
  catNumOptionsLst.pop(maxInd)
  sp0 = np.delete(sp0, maxInd, axis=1)
  sp1 = np.delete(sp1, maxInd, axis=1)
  #recursively call the function on the first split
  ID3(sp0,catNumOptionsLst.copy(), sp0C1, sp0C0, tree, level + 1)
  #recursively call the function on the second split
  ID3(sp1,catNumOptionsLst.copy(), sp1C1, sp1C0, tree, level + 1)

  return tree

#helper function for calculating entropy
def entropy(class0, class1):
  if class0  == 0:
    return 0
  if class1 == 0:
    return 0
  p0 = class0/ (class0 + class1)
  p1 = class1 / (class0 + class1)
  ent = -(p0 * math.log(p0, 2) + p1 * math.log(p1, 2))
  return ent

#helper funciton for calculating information gain
def informationGain(origEntro, split1, split2, class0s1, class0s2):
  info = origEntro - (split1/(split1 + split2) * entropy(class0s1, (split1 - class0s1)) + split2/(split1 + split2) * entropy(class0s2, (split2 - class0s2)))
  return info

#This module takes a given ID3 tree and determine the value of a single input
def ID3Classify(input, tree):
  skip = -1
  lvl = 0
  skCount = 0
  for node in tree:
    lvl = node[1]
    catNum = node[0]

    if skip != -1:
      if lvl == skip:
        skCount = skCount + 1
      if skCount == 2:
        skCount = 0
        skip = -1
        if catNum == -1:
          return 0
        if catNum == -2:
          return 1
        catVal = input[catNum]
        if catVal == 1:
          skip = lvl + 1
        else:
          input = np.delete(input, catNum)
    else:
      if catNum == -1:
        return 0
      if catNum == -2:
        return 1
      catVal = input[catNum]
      if catVal == 1:
        skip = lvl + 1
      else:
        input = np.delete(input, catNum)
  return -1

#converts input data into correct format for ID3 algorithm
def inputConversion(input):
  binInput = np.zeros(34)
  #feature 1 extrapolation
  fin = 0
  for i in range(15):
    if i == input[0]:
      binInput[i] = 1
      fin = 1
  if fin == 0:
    binInput[15] = 1
  #feature 2 extrapolation
  fin = 0
  for i in range(8):
    if i == input[1]:
      binInput[16 + i] = 1
      fin = 1
  if fin == 0:
    binInput[24] = 1
  #feature 3 extrapolation
  fin = 0
  for i in range(4):
    if i == input[2]:
      binInput[25 + i] = 1
      fin = 1
  if fin == 0:
    binInput[29] = 1
  #feature 6 extrapolation
  fin = 0
  for i in range(2):
    if i == input[5]:
      binInput[30 + i] = 1
      fin = 1
  if fin == 0:
    binInput[32] = 1
  return binInput

if __name__ == '__main__':
    app.run(debug=True)
