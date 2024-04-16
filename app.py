from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    # You need to create an 'about.html' template in your templates directory.
    return render_template('about.html')

@app.route('/howitworks')
def how_it_works():
    # You need to create a 'howitworks.html' template in your templates directory.
    return render_template('howitworks.html')

@app.route('/check_domain', methods=['POST'])
def check_domain():
    data = request.get_json()
    domain = data.get('domain', 'No domain provided')
    print(domain)
    # Implement your domain checking logic here
    return jsonify(message=f"Checking safety for domain: {domain}")

if __name__ == '__main__':
    app.run(debug=True)
