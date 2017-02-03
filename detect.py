import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plt

from tsd.network import network
from tsd.data import FlexibleImageDataset
import numpy
import math

import chainer
from chainer import cuda, serializers, Variable
import chainer.functions as F
import chainer.links as L

from PIL import Image
from train import load_config

from sklearn.metrics import confusion_matrix
import itertools

def load_image(filename, size):
    f = Image.open(filename).convert("RGB")
    f = f.resize(size, Image.ANTIALIAS)
    try:
        image = numpy.asarray(f, dtype=numpy.float32)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()

    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, numpy.newaxis]
    
    return image.transpose(2, 0, 1)

def predict_label(image, model):
    ret = model.predictor(numpy.array([image]))
    ret = F.softmax(ret)
    #print ret.data
    lbl = numpy.argmax(ret.data)
    return lbl
"""
def confusion_matrix(result, size):

    matrix = numpy.zeros((size,size))

    for pair in result:
        matrix[pair[0], pair[1]] += 1

    plt.matshow(matrix, fignum=100, cmap=plt.cm.Blues)
    plt.show()
"""
def rms(a, b):
    math.sqrt(a*a + b*b)

def normalize_confusion_matrix(matrix):
    for x in matrix.shape[0]:
        for y in matrix.shape[1]:
            matrix[x,y] /= rms(matrix[x,x], matrix[y,y])
    return matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    config = load_config("../config/gtsrb.json")
    imgsize = (config["size"], config["size"])

    num_classes = config["num_classes"]
    model = L.Classifier(network(1024, num_classes))
    serializers.load_npz("/mnt/sakuradata2/calland/scratch/gtsrb/oversampled/model_epoch_20", model)
    model.predictor.train = False
    
    #model.to_gpu()

    val_annotations = config["validation_annotation"]
    validation_data = FlexibleImageDataset(val_annotations, mean=config["validation_mean"], size=imgsize, normalize=True)
    validation_data.summary()

    #img = load_image("/mnt/sakuradata2/datasets/GTSRB/Final_Test/Images/00001.ppm.png", imgsize)
    #print "predicting image "
    #label = predict_label(img, model)
    #print label

    #exit()
    
    lbl_pred = []
    lbl_true = []

    #matrix = numpy.zeros((num_classes, num_classes))
    class_names = [str(i) for i in range(num_classes)]

    for i in range(len(validation_data._pairs)):
        if i % 100 == 0:
            print "{} / {}".format(i, len(validation_data._pairs) )

        #print validation_data._pairs[i][0]
        data = validation_data.get_example(i)
        arr = numpy.array([data[0]])
        lbl = numpy.array([data[1]])

        #print data[0].dtype
        #print arr.shape
        #print model.predictor.shape
        ret = model.predictor(arr)
        #print ret.data
        ret = F.softmax(ret)
        #print ret.data
        
        lbl_true.append(lbl.tolist()[0])
        lbl_pred.append(numpy.argmax(ret.data))
        #matrix[true, recon] += 1

        #matrix = normalize_confusion_matrix(matrix)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(lbl_true, lbl_pred)
    numpy.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()

    #plt.matshow(matrix, fignum=100, cmap=plt.cm.Blues)
    #plt.show()


if __name__=="__main__":
    main()
