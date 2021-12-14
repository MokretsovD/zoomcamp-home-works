# Car price Prediction

## Description

I decided to choose Malaria Cell Images dataset for my capstone project because I believe it is great for applying what we learned so far about neural networks and deep learning on practice. The goal here is to experiment with different techniques we covered throughout the course and try to get as best result as possible. 

In order to test the project this dataset has to be downloaded from Kaggle by following this [Link](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) and pressing __*Download*__ button at the top right part of the page. Alternatively the same file can be downloaded from [OneDrive](https://1drv.ms/u/s!Ak1DnqTabSUj-PgPn39945WOZoWkyA?e=Qc3K9y) till 01.03.2022. Downloaded 'archive.zip' file must be placed in the same directory as this file

## Files

This project contains several files:
* README.md - this file
* notebook.ipynb - jupyter lab notebook with data preparation, EDA, neural network model tuning and evaluation. Jupyter lab or VSCode + Jupyter extension can be used to open and run it.
    
  __*Warning:*__ running the whole notebook might take very long time if not using graphics card (> 60 minutes, depending on hardware)
* train.py - python script to train, evaluate and save the model
* malaria-model.tflite - final malaria prediction model
* predict.py - web-service to evaluate malaria cells images and predict if they are infected or not
* Pipfile - file with pipenv dependencies
* Pipfile.lock - pipenv file to run predict.py and train.py locally
* Dockerfile - file to dockerize the app
* .dockerignore - docker ignore file
* test-uninfected-image.png - image of uninfected cell to test the model
* test-infected-image.png - image of infected cell to test the model

## How to run train.py and predict.py locally

To guarantee that train and predict files executed against the same versions of libraries pipenv is used. All following commands should be executed from within the *capstone-project-12* folder

First pipenv should be installed. This command should be executed in order to do so:
```sh
pip install pipenv
```

Then required packages should be installed. To do that the following command should be executed:
```sh
pipenv install --dev
```

Now virtual environment is ready and scripts can be run.

To train model:
```sh
pipenv run python train.py
```

This file will do the next things: 
* clean up any existing folders from previous runs 
* extract archive.zip
* rename files and do split train test images
* train the model
* evaluate the model against test dataset
* convert the model to tflite and save it as malaria-model.tflite, if file exists - it will be replaced

To run web app
```sh
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

App implemented using flask framework. It can be accessed on http://localhost:9696/predict via postman, thunderbolt or any other rest client

In order to get a prediction, http request should be sent to localhost

```
POST http://localhost:9696/predict
```
with body, containing url of image to evaluate

```json
{
     "url": "https://github.com/MokretsovD/zoomcamp-home-works/raw/main/capstone-project-12/test-infected-image.png"
 } 
```

the response will contain the probability of looking at healthy cell

```json
{
  "healthy": 0.00022661685943603516
}
```

Two example files are included in the project:
* Use https://github.com/MokretsovD/zoomcamp-home-works/raw/main/capstone-project-12/test-infected-image.png to evaluate infected cell
* Use https://github.com/MokretsovD/zoomcamp-home-works/raw/main/capstone-project-12/test-uninfected-image.png to evaluate uninfected cell


## Docker

T his project contains a Dockerfile. In order to build a container the command should be executed in terminal:

```sh
docker build -t malaria-prediction .
```

Docker should be installed in the system. Please follow the [official instructions](https://docs.docker.com/get-docker/) for your system

To run the container the following command can be used:

```sh
docker run -it -p 9696:9696 malaria-prediction:latest
```

It can be accessed in the same way using REST client at http://localhost:9696/predict

## Deployment

This app can be easily deployed to the cloud of choice using docker file. This part is not covered in the project