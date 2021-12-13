#!/usr/bin/env python
# coding: utf-8

import zipfile
import os
import shutil
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_to_zip_file = './archive.zip'
data_path = './data'
train_path = './train'
val_path = './validation'
test_path = './test'
infected = '/parasitized'
uninfected = '/uninfected'
tflite_model_name = 'malaria-model.tflite'

def rename_files(folder):   
    print('Giving consequent names to files in folder %s' % folder)
    lastCount = 0
    for count, filename in enumerate(os.listdir(folder)):
        extension = filename.split('.')[1]
        dst = f"{str(count)}.{extension}"
        src =f"{folder}/{filename}" 
        dst =f"{folder}/{dst}"
        os.rename(src, dst)
        lastCount = count
    print("File count in folder %s is %s" % (folder, lastCount))

def move_files(extention, min, max, source_folder, dest_folder):
    print('Moving files with extention %s from %s to %s' % (extention, source_folder, dest_folder))
    for file_name in os.listdir(source_folder):
        basename, ext = file_name.split('.')    

        if ext != extention:
            print("file name %s does not have an extention %s" % (file_name, extention))
            continue

        try:
            number = int(basename)
        except ValueError:
            print("Name %s is not numeric" % basename)
            continue  # not numeric

        if (number < min or number > max):
            continue

        source = source_folder + '/' + file_name
        destination = dest_folder + '/'  + file_name
        
        if os.path.isfile(source):
            shutil.move(source, destination)

def ensure_dir(path):
    if not os.path.exists(path):        
        os.makedirs(path)    
        print("Created folder %s" % path)

def validate_files(path = [], extensions = []):
    for folder_path in path:
        print("Path: %s" % folder_path)
        for fldr in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, fldr)
            print('Checking: %s ' % fldr)
            for filee in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, filee)
               
                try:
                    im = Image.open(file_path)
                    rgb_im = im.convert('RGB')
                    if filee.split('.')[1] not in extensions:
                        extensions.append(filee.split('.')[1])
                except:
                    print("Wrong format file: %s" % file_path)
                    os.remove(file_path)        
        
def clean_up_paths(paths_to_cleanup = []):
    for p in paths_to_cleanup:
        if os.path.exists(p): 
            print('Detected folder %s. Cleaning it up.' % p)   
            shutil.rmtree(p)

def make_model():
    inputs = keras.layers.Input(shape=(150,150,3))

    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(6,6), activation='relu')(inputs)
    
    pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(pool1)

    pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)

    vectors = keras.layers.Flatten()(pool2)

    dense = keras.layers.Dense(units=64, activation='relu')(vectors)

    outputs = keras.layers.Dense(1,activation='sigmoid')(dense)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])    
    
    return model            

print('Started data preparation')

paths_to_cleanup = [
    data_path,
    train_path,
    val_path,
    test_path
]

clean_up_paths(paths_to_cleanup)

if (os.path.exists(path_to_zip_file) == False):
    raise Exception('File %s not found. Please put archive with dataset in the same folder as this script' % path_to_zip_file) 

print('Extracting data from %s' % path_to_zip_file)

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(data_path)

print('Giving files consistent names')
rename_files(data_path + "/cell_images/Parasitized")
rename_files(data_path + "/cell_images/Uninfected")

print('Creating directories for train test split')
ensure_dir(train_path + infected)
ensure_dir(train_path + uninfected)
ensure_dir(val_path + infected)
ensure_dir(val_path + uninfected)
ensure_dir(test_path + infected)
ensure_dir(test_path + uninfected)

print('Moving files in train, val, and test')
move_files("png", 11024, 13779, data_path + "/cell_images/Parasitized", test_path + infected)
move_files("png", 11024, 13779, data_path + "/cell_images/Uninfected", test_path + uninfected)

move_files("png", 8267, 11023, data_path + "/cell_images/Parasitized", val_path + infected)
move_files("png", 8267, 11023, data_path + "/cell_images/Uninfected", val_path + uninfected)

move_files("png", 0, 8266, data_path + "/cell_images/Parasitized", train_path + infected)
move_files("png", 0, 8266, data_path + "/cell_images/Uninfected", train_path + uninfected)

print('Removing data path')
clean_up_paths([data_path])

print('Validating files')
validate_files([val_path, test_path, train_path], ['png'])

print('Started training the model')

print('Creating datasets')
seed = 42
batch_size = 32

train_generator = ImageDataGenerator(rescale=1./255)

train_ds = train_generator.flow_from_directory(
    train_path,
    target_size=(150,150),
    class_mode='binary',
    batch_size=batch_size,
    seed = seed
)

validation_generator = ImageDataGenerator(rescale=1./255)

val_ds = validation_generator.flow_from_directory(
    val_path,
    target_size=(150,150),
    class_mode='binary',
    batch_size=batch_size,
    seed = seed,
    shuffle=False
)

test_generator = ImageDataGenerator(rescale=1./255)

test_ds = validation_generator.flow_from_directory(
    test_path,
    target_size=(150,150),
    class_mode='binary',
    batch_size=batch_size,
    seed=seed,
    shuffle=False
)

print('Making the model')
model = make_model()

print('Training the model')
history = model.fit(train_ds, epochs=5, validation_data=val_ds)

print('Evaluating the model')
result = model.evaluate(test_ds)

print('Converting the model to tflite')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print('Saving the converted model')

with open(tflite_model_name,'wb') as f_out:
    f_out.write(tflite_model)