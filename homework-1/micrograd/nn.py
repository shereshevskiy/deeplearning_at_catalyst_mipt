import numpy as np

from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module
        self.bias = bias
        self.W = Tensor(np.random.randn(in_features, out_features))  # note: Value(data[i, j]) was executed in Tensor
        if self.bias:
            self.b = Tensor(np.zeros((1, out_features)))

    def forward(self, inp):
        """Y = W * x + b"""
        if not isinstance(inp, Tensor):
            inp = Tensor(inp)
        out = inp.dot(self.W)
        if self.bias:
            out += self.b
        return out

    def parameters(self):
        # return 1-d list of all parameters List[Value]
        return self.W.parameters() + self.b.parameters()


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        if not isinstance(inp, Tensor):
            inp = Tensor(inp)
        return Tensor(np.maximum(inp.data, Value(0)))  # работает, проверял


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        def softmax(z):
            z -= z.max(1).reshape(-1, 1)
            return z.exp() / z.exp().sum(1).reshape(-1, 1)

        return -np.log(softmax(inp)[np.arange(len(label)), label]).mean()
