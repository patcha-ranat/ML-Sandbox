#!/bin/bash

mkdir data

curl -L -o data/transactions-fraud-datasets.zip https://www.kaggle.com/api/v1/datasets/download/computingvictor/transactions-fraud-datasets

unzip data/transactions-fraud-datasets.zip -d data/

rm data/transactions-fraud-datasets.zip