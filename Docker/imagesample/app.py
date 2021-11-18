import flask
from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    return f"{os.getenv('language')} is an {os.getenv('type')} programming language!!!!"


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000) 