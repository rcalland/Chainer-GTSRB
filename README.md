# chainer-GTSRB
Traffic sign classification on [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=about) using [Chainer](http://chainer.org/).

## Setup

### Download data
```
$ python tools/setup_dataset.py -o $DATA_PATH
```
where `$DATA_PATH` is the path where you would like to download the data files to. The directory will be created if it doesn't already exist. You can specify the URL using the optional `-i` flag.

### Preprocess data
```
$ python tools/oversampler.py
```