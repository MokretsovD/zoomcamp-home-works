FROM agrigorev/zoomcamp-model:3.8.12-slim
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "./"]
RUN sed -i "s|model_file = 'model1.bin'|model_file = 'model2.bin' |g" ./predict.py
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]