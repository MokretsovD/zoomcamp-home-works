from flask import Flask
from flask import request
from flask import jsonify
import pickle
import numpy as np
from datetime import datetime

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('predict')

# car = {
#     "mileage": 235000,
#     "make": "bmw"
#     "model": 316
#     "fuel": "diesel"
#     "gear": "manual"

#     "hp": 116.0
#     "year": 2011 #age=10
# }  # "price": 8.824825 ~ 6800.00						

@app.route('/predict', methods=['POST'])
def predict():
    car = request.get_json()
    car['age'] = datetime.now().year - car['year']

    X = dv.transform([car])
    y_pred = model.predict(X)
    
    price = float(np.expm1(y_pred).round(2))

    result = {
        "estimated_price": price,
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696, reloader_interval=3)