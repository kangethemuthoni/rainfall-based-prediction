import os
import logging
from flask import Flask
from routes import main
from dotenv import load_dotenv

load_dotenv() 

logging.basicConfig(level=logging.DEBUG)

def create_app():
    app = Flask(__name__)

    app.register_blueprint(main)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
