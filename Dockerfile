FROM ubuntu:16.04
MAINTAINER Zheng Zheng <zzmia13@gmail.com>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib scikit-learn

COPY . /
CMD ["python3", "./script.py"]
