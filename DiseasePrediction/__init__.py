from flask import Flask
from .views import DiseasePrediction_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(DiseasePrediction_blueprint)

    return app
