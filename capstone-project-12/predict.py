from flask import Flask
from flask import request
from urllib.request import urlopen
from io import BytesIO
from flask import jsonify
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

model_file = 'malaria-model.tflite'

interpreter = tflite.Interpreter(model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def process_image(img):
  x = np.array(img)
  X = np.array([x/255], dtype='float32')

  return X

def get_image_from_url(url):
  img = download_image(url)
  img = prepare_image(img, (150,150))
  return process_image(img)

app = Flask('predict')  

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()        
    url = body['url']
    print(url)
    X = get_image_from_url(url) 

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()

    return jsonify(float_predictions)

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=9696, reloader_interval=3)