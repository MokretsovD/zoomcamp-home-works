# Car price Prediction

## Description

Sometimes we want to sell our car and the task of choosing the right price for it might be tough. In this is one of many cases when we can use ML to make our life easier. In this project we will use [AutoScout24 dataset from Kaggle](https://www.kaggle.com/ander289386/cars-germany) with data from Germany in order to train several models and select the best one. Also we will create a simple app that allows us to predict price of our car and help us to sell it in this way.

## Files

This project contains several files:
* README.md - this file
* notebook.ipynb = jupyter lab notebook with data cleaning, EDA and several trained models. Jupyter lab or VSCode + Jupyter extension with conda 4.10.3 can be used to open and run it. __*Warning:*__ running the whole notebook might take long time (> 10 minutes, depending on hardware)
* autoscout24-germany-dataset.csv - dataset with car prices we are suing to train our model
* train.py - python script to train and save model usind dataset
* model.bin - the best selected model to predict our price
* predict.py - web-service to predict prices
* Pipfile.lock - virtual environment file to run predict.py and train.py easier locally
* Dockerfile - file to dockerize our app

## How to run train.py and predict.py locally

To guarantee that train and predict files executed against the same versions of libraries pipenv is used. First it should be installed.

Run this command in terminal to install pipenv
```sh
pip install pipenv
```
Then required packages should be installed. To do that, in terminal, when in midterm-project-07 folder run the command
```sh
pipenv install
```

Now virtual environment is ready and files can be run.

To train model:
```sh
pipenv run python train.py
```

and to run web app
```sh
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

App then can be accessed on http://localhost:9696/predict via postman or any toher rest client

In order to get a prediction use client of your choice do

```
POST http://localhost:9696/predict
```
with body like

```json
 {
     "mileage": 235000,
     "make": "bmw",
     "model": 316,
     "fuel": "diesel",
     "gear": "manual",
     "hp": 116.0,
     "year": 2011
 } 
```

the response will return price prediction

```json
{
  "estimated_price": 5821.55
}
```

## Docker

Project also contains a Dockerfile. In order to build a container run the command in terminal:

```sh
docker build -t car-price-prediction .
```

Docker should be installed in the system. Please follow the [official instructions](https://docs.docker.com/get-docker/) for your system

To run the container run command:

```sh
docker run -it -p 9696:9696 car-price-prediction:latest
```

It can be accessed in the same way using REST client at http://localhost:9696/predict


## Deployment

This app can be easily deployed to the cloud of choice. There are some very nice instructions from our great fellows on how to deploy to:

* [AWS Elastic Beanstalk (Thnx Alexey G.)](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/07-aws-eb.md)
* [PythonAnywhere (Thnx nindate)](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-pythonanywhere.md)
* [Heroku (Thnx razekmaiden)](https://github.com/razekmaiden/churn_service_heroku)