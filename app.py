from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import HashingVectorizer
import string

app = Flask(__name__)

# Define the base directory and static folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(BASE_DIR, 'static')
valid_domains_csv_path = os.path.join(static_folder, 'valid-domains.csv')
invalid_domains_csv_path = os.path.join(static_folder, 'invalid-domains.csv')

def load_and_prepare_data(valid_domains_csv, invalid_domains_csv):
    valid_domains = pd.read_csv(valid_domains_csv, header=None)
    invalid_domains = pd.read_csv(invalid_domains_csv, header=None)
    valid_domains[1] = 1
    invalid_domains[1] = 0
    combined_data = pd.concat([valid_domains, invalid_domains], axis=0)
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    X = combined_data[0]
    y = combined_data[1]
    return X, y

# Feature extraction functions
def domain_features(domain):
    features = {
        'length': len(domain),
        'digit_count': sum(char.isdigit() for char in domain),
        'letter_count': sum(char.isalpha() for char in domain),
        'special_char_count': sum(char in set(string.punctuation) for char in domain),
    }
    return features

# Create numeric features from domain names
def create_feature_matrix(domains):
    vectorizer = HashingVectorizer(n_features=10)
    domain_vector = vectorizer.transform(domains).toarray()
    additional_features = np.array([list(domain_features(domain).values()) for domain in domains])
    return np.hstack((domain_vector, additional_features))

# Load data
X, y = load_and_prepare_data(valid_domains_csv_path, invalid_domains_csv_path)
X = create_feature_matrix(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howitworks')
def how_it_works():
    return render_template('howitworks.html')

@app.route('/check_domain', methods=['POST'])
def check_domain():
    data = request.get_json()
    domain = data.get('domain', 'No domain provided')
    # Generate features for the provided domain
    domain_features_vector = create_feature_matrix(np.array([domain]))
    prediction = classifier.predict(domain_features_vector)
    status = 'valid' if prediction[0] == 1 else 'invalid'
    return jsonify(message=f"Checking safety for domain: {domain}", status=status)

if __name__ == '__main__':
    app.run(debug=True)
