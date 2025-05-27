import os
import logging
import pickle
import numpy as np
import jwt
from datetime import datetime, timedelta
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from detector.feature import FeatureExtraction
import re
from dotenv import load_dotenv

# Get config
def get_origins():
    import json
    with open("origins.json", 'r') as file:
        data = json.load(file)
    return data.get('origins', [])

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, supports_credentials=True, origins=get_origins(), allow_headers=["Authorization", "Content-Type", "authToken"])

# MongoDB URI and Secret Key
load_dotenv()
app.config['MONGO_URI'] = os.getenv("MONGO_URI")
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

mongo = PyMongo(app)
bcrypt = Bcrypt(app)

# Logging Configuration
logging.basicConfig(
    filename="logs/app.log",
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Load the model
try:
    with open("model/newmodel.pkl", "rb") as file:
        gbc = pickle.load(file)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file not found.")
    raise

# JWT Utility Functions
def generate_token(username, email, profile_pic=None):
    return jwt.encode(
        {"username": username, "email": email, "profileImage": profile_pic, "exp": datetime.utcnow() + timedelta(days=28)},
        app.config['SECRET_KEY'], algorithm='HS256'
    )

def verify_token(token):
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        logging.info(f"Token verified for user: {decoded.get('username')}")
        return decoded.get('username')
    except jwt.ExpiredSignatureError:
        logging.warning("Token expired.")
    except jwt.InvalidTokenError:
        logging.warning("Invalid token.")
    return None

# URL Prediction Function
def predict(url):
    features = FeatureExtraction(url).getFeaturesList()
    prediction = bool(gbc.predict(np.array(features).reshape(1, -1))[0])
    logging.info(f"Prediction result for {url}: {prediction}")
    return prediction

def validate_email(email):
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(regex, email))

def validate_password(password):
    if len(password) < 8:
        return False
    return True

def validate_username(username):
    if len(username) < 3 or len(username) > 20:
        return False
    if not re.match(r'^[a-z0-9_]+$', username):
        return False
    return True

# Global Error Handling
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unexpected error: {e}")
    return jsonify({"error": "An unexpected error occurred."}), 500

@app.before_request
def log_request():
    logging.info(f"Incoming request: {request.method} {request.url} - Headers: {dict(request.headers)}")

# API Endpoints
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/v1/register', methods=['POST'])
def register_user():
    data = request.get_json()
    username, password, email, profileImage = data.get('username'), data.get('password'), data.get('email'), data.get('profileImage')

    if not (username and password and email):
        logging.warning("Missing username, password, or email")
        return jsonify({"error": "Missing username, password, or email"}), 400

    if not validate_email(email):
        logging.warning("Invalid email format")
        return jsonify({"error": "Invalid email format"}), 400
    
    if not validate_username(username):
        logging.warning("Invalid username format")
        return jsonify({"error": "Invalid username format"}), 400
    
    if not validate_password(password):
        logging.warning("Invalid password format")
        return jsonify({"error": "Invalid password format"}), 400

    if mongo.db.users.find_one({"username": username}):
        logging.warning(f"User {username} already exists")
        return jsonify({"error": "User already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    mongo.db.users.insert_one({"username": username, "password": hashed_password, "email": email, "profileImage": profileImage})
    logging.info(f"User registered successfully: {username}")
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/v1/login', methods=['POST'])
def login_user():
    data = request.get_json()
    user = mongo.db.users.find_one({"username": data.get('username')})

    if user and bcrypt.check_password_hash(user['password'], data.get('password')):
        return jsonify({
            "token": generate_token(user['username'], user['email'], user['profileImage']),
            "username": user['username']
        }), 200

    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/v1/auth', methods=['GET'])
def authenticate_user():
    username = verify_token(request.headers.get('authToken'))
    if username:
        logging.info(f"User authenticated: {username}")
        return jsonify({"message": "Authenticated successfully", "username": username}), 200
    logging.warning("Invalid or expired token")
    return jsonify({"error": "Invalid or expired token"}), 401

@app.route('/api/v1/check', methods=['POST'])
def check_url():
    username = verify_token(request.headers.get('authToken'))
    if not username:
        logging.warning("Invalid or expired token")
        return jsonify({"error": "Invalid or expired token"}), 401

    url = request.get_json().get('url')
    if not url:
        logging.warning("Missing URL parameter")
        return jsonify({"error": "Missing 'url' parameter"}), 400

    url_data = mongo.db.urls.find_one({"username": username, "url": url})
    if url_data:
        logging.info(f"URL {url} already exists in the database for {username}")
        return jsonify(url_data), 200

    status = predict(url)
    url_data = {"username": username, "domain": urlparse(url).netloc, "url": url, "is_safe": status, "is_blocked": not status}
    mongo.db.urls.insert_one(url_data)

    logging.info(f"URL check completed for {url}, Safe: {status}")
    return jsonify(url_data), 200

@app.route('/api/v1/block-toggle', methods=['POST'])
def unblock_url():
    username = verify_token(request.headers.get('authToken'))
    if not username:
        return jsonify({"error": "Invalid or expired token"}), 401

    url = request.get_json().get('url')
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400
    
    url_data = mongo.db.urls.find_one({"username": username, "url": url})
    new_status = not url_data.get('is_blocked')
    mongo.db.urls.update_one({"username": username, "url": url}, {"$set": {"is_blocked": new_status}})

    logging.info(f"Block status toggled for {url}, Blocked: {new_status}")
    return jsonify({"message": "Block status changed successfully."}), 200

@app.route('/api/v1/delete-account', methods=['POST'])
def delete_account():
    username = verify_token(request.headers.get('authToken'))
    if not username:
        return jsonify({"error": "Invalid or expired token"}), 401

    mongo.db.users.delete_one({"username": username})
    mongo.db.urls.delete_many({"username": username})

    logging.info(f"User account deleted: {username}")
    return jsonify({"message": "Account deleted successfully"}), 200

@app.route('/api/v1/urls', methods=['GET'])
def get_urls():
    username = verify_token(request.headers.get('authToken'))
    if not username:
        return jsonify({"error": "Invalid or expired token"}), 401

    urls = list(mongo.db.urls.find({"username": username}, {"_id": 0}))
    logging.info(f"Retrieved URLs for user: {username}")
    return jsonify(urls), 200

@app.route('/api/v1/delete-url', methods=['POST'])
def delete_url():
    username = verify_token(request.headers.get('authToken'))
    if not username:
        return jsonify({"error": "Invalid or expired token"}), 401

    url = request.get_json().get('url')
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    result = mongo.db.urls.delete_one({"username": username, "url": url})
    if result.deleted_count == 0:
        logging.warning(f"URL {url} not found for user {username}")
        return jsonify({"error": "URL not found"}), 404

    logging.info(f"URL {url} deleted successfully for user {username}")
    return jsonify({"message": "URL deleted successfully"}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')