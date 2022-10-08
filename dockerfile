FROM python:3.10
LABEL Maintainer="Calvin Joseph"
COPY . /emotionprediction
WORKDIR /emotionprediction
RUN pip install -r requirements.txt
CMD [ "python" ,"bert_prediction.py"]