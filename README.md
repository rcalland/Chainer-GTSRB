# chainer-GTSRB
Traffic sign classification on [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=about) using [Chainer](http://chainer.org/). The model closely follows [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

## Setup
If you want to get going fast, just do `cd tools; source deploy.sh $DATA_PATH` to get everything done. If you need to tweak any of the data processing steps, follow below.

### Download data
```
$ python tools/setup_dataset.py -o $DATA_PATH
```
where `$DATA_PATH` is the path where you would like to download the data files to. The directory will be created if it doesn't already exist. You can specify the URL using the optional `-i` flag.

### Preprocess data
Please move into the tools directory before running these scripts.
```
cd tools
$ python oversampler.py
$ python make_annotations.py
$ python generate_means.py
```

### Training & Detection
```
$ python train.py
```

```
$ python detect.py
```