# Reference: https://hub.docker.com/_/python
FROM python:3.10-bullseye
WORKDIR /app

# Install needed packages
RUN pip install numpy pandas
RUN pip install matplotlib seaborn
RUN pip install tensorflow
RUN pip install scikit-learn