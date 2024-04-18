from flask import Flask, render_template, request, jsonify
import string
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import math
import pandas as pd

app = Flask(__name__)

# Global variable for the model
model = None

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
    data = request.get_json()
    domain = data.get('domain', 'No domain provided')
    print(domain)
    process_domain(domain)
    return jsonify(message=f"Checking safety for domain: {domain}")

# Processess the domain for feature extraction and prediction
def process_domain(domain):
    domain_data = calculate_domain_characteristics(domain)
    validity = predict_domain_validity(domain)
    write_to_csv(domain_data, validity)

# Calculates the domain features
def calculate_domain_characteristics(domain):
    characteristics = {
        'domain': domain,
        'domain_length': len(domain),
        'domain_char_count': sum(c.isalpha() for c in domain.split('.')[0]),
        'domain_digit_count': sum(c.isdigit() for c in domain.split('.')[0]),
        'repeated_chars': 1 if has_repeated_chars(domain.split('.')[0], 'alpha') else 0,
        'repeated_digits': 1 if has_repeated_chars(domain.split('.')[0], 'digit') else 0,
        'non_ascii_char_count': count_non_ascii_chars(domain.split('.')[0]),
        'domain_tld': domain.split('.')[-1] if '.' in domain else ''
    }
    return characteristics

# Helper to determine if domain has repeated characters before TLD
def has_repeated_chars(s, char_type):
    return any(s[i] == s[i+1] for i in range(len(s) - 1) if (s[i].isdigit() if char_type == 'digit' else s[i].isalpha()))

# Helper to determine if domain has non-ascii characters before TLD
def count_non_ascii_chars(s):
    return sum(1 for c in s if ord(c) > 127)

# Write's the domain + calculated characteristics to the respective csv file
def write_to_csv(domain_data, validity):
    filename = 'static/valid-domains.csv' if validity == 'valid' else 'static/invalid-domains.csv'
    fieldnames = list(domain_data.keys())
    write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(domain_data)

#train model
def train_model():
    global model
    #concatenate the two dataframes
    df1 = pd.read_csv('static/valid-domains.csv')
    df2 = pd.read_csv('static/invalid-domains.csv')
    df1['validity'] = 1
    df2['validity'] = 0
    df = pd.concat([df1, df2])

    #prepare the data
    X = df[['domain_length', 'domain_char_count','domain_digit_count','repeated_chars','repeated_digits', 'non_ascii_char_count']]
    y = df['validity']

    #split the dataset into training set and test set
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    #create a DecisionTreeClassifier object
    model = DecisionTreeClassifier()

    #train the model using the training sets
    model.fit(XTrain, yTrain)

# Implement ML modle here given the domain, based on prediction be sure to return 'valid' or 'invalid'
def predict_domain_validity(domain):

    #return 'valid' if sum(c.isalpha() for c in domain) > sum(c.isdigit() for c in domain) else 'invalid'
    global model
    # ensure the model is trained
    if model is None:
        return 'Model not trained'

    #get the characteristics of the domain
    domain_data = calculate_domain_characteristics(domain)
    #remove 'domain' and 'domain_tld' from the dictionary as not needed in model
    domain_data.pop('domain', None)
    domain_data.pop('domain_tld', None)

    #convert to DataFrame
    domain_df = pd.DataFrame([domain_data])

    #predict the validity
    prediction = model.predict(domain_df)

    #return 'valid' if prediction is 1, 'invalid' otherwise
    return 'valid' if prediction[0] == 1 else 'invalid'

if __name__ == '__main__':
    #train model before running app
    train_model()
    app.run(debug=True)
