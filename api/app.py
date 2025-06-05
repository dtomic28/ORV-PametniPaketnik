import uuid
from flask import Flask, request, jsonify, render_template, Response
import os, zipfile, threading, io
import sys
# Add project root (parent of 'api') to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_recognition.predict import predict_image
from face_recognition.raw_image_processing import preprocess
from face_recognition.train import train_binary_model
from log_stream import log_stream, StreamToLogger

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redirect stdout and stderr to custom stream for logging
#sys.stdout = sys.stderr = StreamToLogger()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logs')
def logs():
    return Response(log_stream(), mimetype='text/event-stream')

@app.route('/upload/<username>', methods=['POST'])
def upload_images(username):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    zip_file = request.files['file']
    if not zip_file.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a .zip archive'}), 400

    target_dir = os.path.join("..", "face_recognition", "images", username, "originals")
    os.makedirs(target_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
            z.extractall(target_dir)
        print(f"[upload] Uploaded and extracted images for '{username}' to: {target_dir}")
        return jsonify({'message': 'Upload successful.'})
    except Exception as e:
        print(f"[upload][error] {str(e)}")
        return jsonify({'error': 'Extraction failed.'}), 500

@app.route('/train/<username>', methods=['POST'])
def train_user(username):
    thread = threading.Thread(target=run_pipeline, args=(username,))
    thread.start()
    print(f"[train] Training pipeline started for user: {username}")
    return jsonify({'message': f'Training started for {username}'}), 200

def run_pipeline(username):
    try:
        print(f"[pipeline] Preprocessing for {username}")
        preprocess(username)
        print(f"[pipeline] Training for {username}")
        train_binary_model(username)
        print(f"[pipeline] Completed for {username}")
    except Exception as e:
        print(f"[pipeline][error] {str(e)}")

@app.route('/predict/<username>', methods=['POST'])
def predict_route(username):
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save image temporarily
    os.makedirs("tmp", exist_ok=True)
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("tmp", temp_filename)
    image_file.save(temp_path)

    # Resolve model path
    model_path = os.path.join("..", "face_recognition", "models", f"{username}_model.h5")
    print(f"Model path: {model_path}")
    try:
        prediction = predict_image(temp_path, model_path)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)