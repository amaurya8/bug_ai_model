from flask import Flask, request, jsonify
import src.bug_predict_model as bug_predict_model
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']
    predicted_bug_or_error =  bug_predict_model.predict_bug_or_error(input_text)
    response = {'prediction': predicted_bug_or_error}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
