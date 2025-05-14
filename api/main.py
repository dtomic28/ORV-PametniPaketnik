import os
from flask import Flask, request, jsonify
import json

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World\! This is a Flask & MongoDB app deployed on Fly.io"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)  # change to your ip address
