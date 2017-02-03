#!/bin/bash

python setup_dataset.py -o $1
python oversampler.py
python make_training_annotations.py
python generate_means.py

