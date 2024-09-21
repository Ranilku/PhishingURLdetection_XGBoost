from flask import Flask, request, render_template
import pandas as pd
import joblib
from urllib.parse import urlparse
import numpy as np
import whois
import socket
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load(r'C:\urllegitimacy\model\phishing_model_specified_features.joblib')
scaler = joblib.load(r'C:\urllegitimacy\model\scaler_specified_features.joblib')
label_encoders = joblib.load(r'C:\urllegitimacy\model\label_encoders_specified_features.joblib')

# Function to calculate entropy
def calculate_entropy(string):
    import math
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

# Function to extract features from a single URL
def extract_features_from_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Calculate domain age and handle exceptions
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        domain_age = (datetime.now() - creation_date).days / 365 if creation_date else -1
    except:
        domain_info = None
        domain_age = -1

    # Get IP address and handle exceptions
    try:
        ip_address = socket.gethostbyname(domain)
    except socket.gaierror:
        ip_address = '0.0.0.0'
    
    # Function to safely encode a feature, handling unseen labels
    def safe_label_encode(encoder, value, treat_as_phishing=False):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return len(encoder.classes_) if treat_as_phishing else len(encoder.classes_) + 1  # Treat as phishing or assign a new label for unseen value
    
    # Extract features based on what was used in training
    features = {
        'domain_entropy': calculate_entropy(domain),
        'first_token': safe_label_encode(label_encoders['first_token'], domain.split('.')[0]),
        'domain_age': domain_age,
        'frequency_of_slash': path.count('/'),
        'subdomain_length': len(domain.split('.')[0]),
        'ip_address': safe_label_encode(label_encoders['ip_address'], ip_address),
        'average_word_length': np.mean([len(token) for token in path.split('/')]),
        'alphanumeric_ratio': sum(c.isalnum() for c in url) / len(url),
        'tld': safe_label_encode(label_encoders['tld'], domain.split('.')[-1], treat_as_phishing=True),
        'registrar': safe_label_encode(label_encoders['registrar'], domain_info.registrar if domain_info and domain_info.registrar else 'Unknown'),
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url),
        'https_domain': 1 if parsed_url.scheme == 'https' else 0,
        'url_path_length': len(path),
        'longest_word_length': max([len(token) for token in path.split('/')], default=0),
        'frequency_of_hyphen': url.count('-')
    }
    
    return pd.Series(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        show_features = request.form.get('show_features', 'no')
        features = extract_features_from_url(url)
        features_df = pd.DataFrame([features])
        
        # Standardize the features
        features_scaled = scaler.transform(features_df)

        # Predict probabilities
        probabilities = model.predict_proba(features_scaled)
        
        legitimate_prob = probabilities[0][0] * 100
        phishing_prob = probabilities[0][1] * 100
        
        return render_template('index.html', url=url, legitimate_prob=legitimate_prob, phishing_prob=phishing_prob, show_features=show_features, features=features.to_dict())
    
    return render_template('index.html', url=None)

if __name__ == '__main__':
    app.run(debug=True)
