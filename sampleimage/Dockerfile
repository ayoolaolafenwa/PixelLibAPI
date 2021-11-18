FROM python:3.8-slim-buster

ENV language Python

ENV type Object Oriented 

WORKDIR /app

EXPOSE 5000

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]