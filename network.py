import chainer
import chainer.functions as F
import chainer.links as L

class network(chainer.Chain):
    def __init__(self, n_linear, n_out):
        features = [32, 64, 128]

        super(network, self).__init__(
            conv0=L.Convolution2D(None, features[0], 5, stride=1, pad=2),
            conv1=L.Convolution2D(None, features[1], 5, stride=1, pad=2),
            conv2=L.Convolution2D(None, features[2], 5, stride=1, pad=2),
            bn0=L.BatchNormalization(features[0]),
            bn1=L.BatchNormalization(features[1]),
            bn2=L.BatchNormalization(features[2]),
            bn3=L.BatchNormalization(n_linear),
            l0=L.Linear(None, n_linear),
            l1=L.Linear(None, n_out))

        self.train = True

    def __call__(self, x):
        dropout_ratio = [0.1, 0.2, 0.3, 0.5]
        #dropout_ratio = [0.0, 0.0, 0.0, 0.5]
        #dropout_ratio = [0.0, 0.0, 0.0, 0.0]

        # conv layers
        l0a = F.dropout(self.bn0(self.conv0(x), test=not self.train), ratio=dropout_ratio[0], train=self.train)
        l0b = F.max_pooling_2d(F.relu(l0a), 2)

        l1a = F.dropout(self.bn1(self.conv1(l0b), test=not self.train), ratio=dropout_ratio[1], train=self.train)
        l1b = F.max_pooling_2d(F.relu(l1a), 2)

        l2a = F.dropout(self.bn2(self.conv2(l1b), test=not self.train), ratio=dropout_ratio[2], train=self.train)
        l2b = F.max_pooling_2d(F.relu(l2a), 2)

        # mult-scale connections
        ms = F.concat((l2b, F.max_pooling_2d(l1b, 2), F.max_pooling_2d(l0b, 4)))

        # FC layers
        l3a = F.dropout(self.bn3(self.l0(ms), test=not self.train), ratio=dropout_ratio[3], train=self.train)
        l3b = F.relu(l3a)

        output = self.l1(l3b)

        return output
