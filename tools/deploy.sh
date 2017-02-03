#!/bin/bash

if [ -z ${1+x} ]; then echo "You need to specify the data path, e.g. source deploy.sh /home/user/data"; return; else echo "Setting up data in ${1}"; fi

#python setup_dataset.py -o $1
python oversampler.py
python make_training_annotations.py
python generate_means.py

echo "All done!"
return
