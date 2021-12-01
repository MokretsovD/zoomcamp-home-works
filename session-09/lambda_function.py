#!/usr/bin/env python
# coding: utf-8
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

interpreter = tflite.Interpreter('cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
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

def predict(url):
    X = get_image_from_url(url) 

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()

    return float_predictions
    
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)

    return result