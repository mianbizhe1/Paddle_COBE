import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
                m.weight.set_value(m.weight * scale)  # for residual block
                if m.bias is not None:
                    nn.initializer.Constant(0.0)(m.bias)
            elif isinstance(m, nn.Linear):
                nn.initializer.KaimingNormal()(m.weight)
                m.weight.set_value(m.weight * scale)
                if m.bias is not None:
                    nn.initializer.Constant(0.0)(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(1.0)(m.weight)
                nn.initializer.Constant(0.0)(m.bias)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Layer):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class ResidualBlock(nn.Layer):
    '''Residual block w. BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.bn = nn.BatchNorm2D(nf)
        self.conv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        return identity + out
