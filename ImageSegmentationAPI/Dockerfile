FROM python:3.8-slim-buster

RUN apt-get -y update && apt-get install -y --fix-missing sudo

RUN apt-get install -y --fix-missing \ 
    libsm6 \ 
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*


RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

EXPOSE 5000

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "segmentation.py"]