import requests

url = 'http://127.0.0.1:5000/predict'
input_data = {'text': 'assertion error'}
response = requests.post(url, json=input_data)

if response.status_code == 200:
    result = response.json()
    predicted_bug_or_error = result['prediction']
    print(f"Predicted Bug or Error: {predicted_bug_or_error}")
