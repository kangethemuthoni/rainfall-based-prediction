from flask import Blueprint, request, jsonify
from utils import preprocess_text

main = Blueprint('main', __name__)

@main.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    processed_text = preprocess_text(text)  

    response = {
        "processed_text": processed_text
    }
    return jsonify(response)
