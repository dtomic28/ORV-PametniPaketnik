import os
from flask import Flask, request, jsonify
import json
import db
import zipfile
from threading import Thread
from ..functions import train, preprocess

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World! This is a Flask & MongoDB app deployed on Fly.io"

@app.route('/upload-zip', methods=['POST'])
def upload_zip():
    if 'username' not in request.form:
        return jsonify({'error': 'Missing username in form data'}), 400

    if 'zipfile' not in request.files:
        return jsonify({'error': 'No zipfile provided'}), 400

    username = request.form['username']
    uploaded_file = request.files['zipfile']

    if not uploaded_file.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a .zip archive'}), 400

    # Define the extraction path
    extraction_path = os.path.abspath(os.path.join('..', 'face_recognition',"images", username, 'originals'))
    os.makedirs(extraction_path, exist_ok=True)

    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        return jsonify({'message': f'Images extracted to {extraction_path}'}), 200
    except zipfile.BadZipFile:
        return jsonify({'error': 'Uploaded file is not a valid ZIP archive'}), 400

def run_training(username):
    try:
        preprocess(username)
        train(username)
    except Exception as e:
        print(f"[ERROR] Training failed for {username}: {e}")

@app.route('/train', methods=['POST'])
def start_training():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Missing username"}), 400

    Thread(target=run_training, args=(username,), daemon=True).start()
    return jsonify({"status": f"Training started in background for {username}"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
