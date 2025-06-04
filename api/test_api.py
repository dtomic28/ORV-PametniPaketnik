import requests
import sys
import os

BASE_URL = 'http://127.0.0.1:5000'

def upload_zip(username):
    url = f'{BASE_URL}/upload/{username}'
    with open('Archive.zip', 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    print(response.status_code, response.json())

def trigger_training(username):
    url = f'{BASE_URL}/train/{username}'
    response = requests.post(url)
    print(response.status_code, response.json())

def predict_image(username, file_path):
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        return
    url = f'{BASE_URL}/predict/{username}'
    with open(abs_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    print(response.status_code, response.json())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test_api.py <username> <file_path|upload|train>")
        sys.exit(1)

    username = sys.argv[1]
    action = sys.argv[2]

    if action == 'upload':
        upload_zip(username)
    elif action == 'train':
        trigger_training(username)
    else:
        predict_image(username, action)
