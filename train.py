from network import network
from data import FlexibleImageDataset

import json
import os
import time
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.functions.loss import softmax_cross_entropy

from tools.utils import load_config

# custom test extension that turns off dropout
class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.predictor.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.predictor.train = True
        return ret

def main():
    config = load_config("config/gtsrb.json")

    # put these as arguments
    gpu_id = 0
    num_epochs = 500
    multiprocess_iterator = True
    n_processes = 4 # only valid if the above is true
    num_classes = config["num_classes"]
    batchsize = config["batchsize"]
    data_annotations = config["train_annotation"]
    val_annotations = config["validation_annotation"]
    output_directory = "/mnt/sakuradata2/calland/scratch/gtsrb/test"
    os.system("mkdir -p {}".format(output_directory))

    if gpu_id >= 0:
        chainer.cuda.get_device(gpu_id).use() # use this GPU

    # fetch training data
    size = (config["size"], config["size"])
    training_data = FlexibleImageDataset(data_annotations, mean=config["train_mean"], size=size)
    validation_data = FlexibleImageDataset(val_annotations, mean=config["validation_mean"], size=size)
    #training_data.summary()

    weights = training_data.get_class_weights(gpu_id)

    predictor = network(config["fc_size"], num_classes)
    loss = softmax_cross_entropy.SoftmaxCrossEntropy(class_weight=weights)

    model = L.Classifier(predictor, lossfun=loss)

    # push to GPU
    if gpu_id >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if config["weight_decay"] > 0.0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(config["weight_decay"]))

    if multiprocess_iterator:
        train_iter = chainer.iterators.MultiprocessIterator(training_data, batchsize, n_processes=n_processes)
        test_iter = chainer.iterators.MultiprocessIterator(validation_data, batchsize, n_processes=n_processes,
                                                           repeat=False, shuffle=False)
    else:
        train_iter = chainer.iterators.SerialIterator(training_data, batchsize)
        test_iter = chainer.iterators.SerialIterator(validation_data, batchsize,
                                                     repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (num_epochs, 'epoch'), out=output_directory)

    val_interval = 1, 'epoch'

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(TestModeEvaluator(test_iter, model,
                                     device=gpu_id), trigger=val_interval)
    #trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=val_interval)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=100))

    #if args.resume:
        # Resume from a snapshot
        #chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
