"""
Altered version (possible Sigmoid in last layer) of deepobs networks
"""
from torch import nn
from deepobs.pytorch.testproblems.testproblems_utils import tfconv2d, \
    tfmaxpool2d, flatten, mean_allcnnc, _truncated_normal_init


class net_cifar100_allcnnc(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    """

    def __init__(self, use_sigmoid=False):
        super(net_cifar100_allcnnc, self).__init__()

        self.add_module('dropout1', nn.Dropout(p=0.2))

        self.add_module('conv1', tfconv2d(in_channels=3, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', tfconv2d(in_channels=96, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3',
                        tfconv2d(in_channels=96, out_channels=96, kernel_size=3, stride=(2, 2), tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())

        self.add_module('dropout2', nn.Dropout(p=0.5))

        self.add_module('conv4', tfconv2d(in_channels=96, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu4', nn.ReLU())
        self.add_module('conv5', tfconv2d(in_channels=192, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu5', nn.ReLU())
        self.add_module('conv6', tfconv2d(in_channels=192, out_channels=192, kernel_size=3, stride=(2, 2),
                                          tf_padding_type='same'))
        self.add_module('relu6', nn.ReLU())

        self.add_module('dropout3', nn.Dropout(p=0.5))

        self.add_module('conv7', tfconv2d(in_channels=192, out_channels=192, kernel_size=3))
        self.add_module('relu7', nn.ReLU())
        self.add_module('conv8', tfconv2d(in_channels=192, out_channels=192, kernel_size=1, tf_padding_type='same'))
        self.add_module('relu8', nn.ReLU())
        self.add_module('conv9', tfconv2d(in_channels=192, out_channels=100, kernel_size=1, tf_padding_type='same'))
        if use_sigmoid:
            self.add_module('sigmoid9', nn.Sigmoid())
        else:
            self.add_module('relu9', nn.ReLU())

        self.add_module('mean', mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


class net_cifar10_3c3d(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    """

    def __init__(self, use_sigmoid=False):
        super(net_cifar10_3c3d, self).__init__()

        self.add_module('conv1', tfconv2d(in_channels=3, out_channels=64, kernel_size=5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('maxpool1', tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(in_channels=64, out_channels=96, kernel_size=3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('maxpool2', tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv3', tfconv2d(in_channels=96, out_channels=128, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())
        self.add_module('maxpool3', tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('flatten', flatten())

        self.add_module('dense1', nn.Linear(in_features=3 * 3 * 128, out_features=512))
        self.add_module('relu4', nn.ReLU())
        self.add_module('dense2', nn.Linear(in_features=512, out_features=256))
        if use_sigmoid:
            self.add_module('relu5', nn.Sigmoid())
        else:
            self.add_module('relu5', nn.ReLU())
        self.add_module('dense3', nn.Linear(in_features=256, out_features=10))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)


class net_fmnist_2c2d(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    """

    def __init__(self, use_sigmoid=False):
        """Args:
            num_outputs (int): The numer of outputs (i.e. target classes)."""

        super(net_fmnist_2c2d, self).__init__()
        self.add_module('conv1', tfconv2d(in_channels=1, out_channels=32, kernel_size=5, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('max_pool1', tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(in_channels=32, out_channels=64, kernel_size=5, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('max_pool2', tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type='same'))

        self.add_module('flatten', flatten())

        self.add_module('dense1', nn.Linear(in_features=7 * 7 * 64, out_features=1024))
        if use_sigmoid:
            self.add_module('relu3', nn.Sigmoid())
        else:
            self.add_module('relu3', nn.ReLU())

        self.add_module('dense2', nn.Linear(in_features=1024, out_features=10))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(module.weight.data, mean=0, stddev=0.05)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(module.weight.data, mean=0, stddev=0.05)
