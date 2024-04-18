from flask import Flask, render_template, request, jsonify
import string

app = Flask(__name__)

# Feature extraction functions
def domain_features(domain):
    features = {
        'length': len(domain),
        'digit_count': sum(char.isdigit() for char in domain),
        'letter_count': sum(char.isalpha() for char in domain),
        'special_char_count': sum(char in set(string.punctuation) for char in domain),
    }
    return features

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
    print(domain)
    # Generate features for the provided domain
    return jsonify(message=f"Checking safety for domain: {domain}")

if __name__ == '__main__':
    app.run(debug=True)
