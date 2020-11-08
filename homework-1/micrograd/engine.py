from __future__ import division

import numpy as np
from typing import Union


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=""):
        self.data = data.data if isinstance(data, Value) else data  # my
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")  # my code

        def _backward():
            self.grad += other.data * out.grad  # my code
            other.grad += self.data * out.grad  # my code

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), "**")  # my code

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad  # my

        out._backward = _backward

        return out

    def exp(self):
        out = Value(np.exp(self.data), (self, ), "exp")  # my code

        def _backward():
            self.grad += out.data * out.grad  # my

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(self.data, 0), (self, ), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            # YOUR CODE GOES
            v._backward()
            self.grad *= v.grad

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __le__(self, other):
        if isinstance(other, Value):
            return self.data <= other.data
        return self.data <= other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.data < other.data
        return self.data < other

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.data > other.data
        return self.data > other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Tensor:
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """
    def __init__(self, data):
        # data elements must be of type Value
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            data = np.array(data)
            self.data = np.array([Value(item) for item in data.flatten()]).reshape(data.shape)

    def __add__(self, other):
        if isinstance(other, Tensor):
            # assert self.shape() == other.shape()  # note: лишнее, ошибку и так даст, если суждено
            return Tensor(np.add(self.data, other.data))
        return Tensor(self.data + other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.multiply(self.data, other.data))
        return Tensor(self.data * other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __floordiv__(self, other):
        return Tensor((self.data ** other.data ** -1).astype(int))
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def exp(self):
        return Tensor(np.exp(self.data))

    def dot(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data)
        return Tensor(self.data @ other)

    def shape(self):
        return self.data.shape

    def argmax(self, dim=None):
        return Tensor(np.argmax(self.data, axis=dim))

    def max(self, dim=None, keepdims=False):
        return Tensor(np.amax(self.data, axis=dim, keepdims=keepdims))

    def sum(self, dim=None):
        return Tensor(np.sum(self.data, axis=dim))

    def mean(self, dim=None):
        return Tensor(np.mean(self.data, axis=dim))

    def reshape(self, *args, **kwargs):
        self.data = self.data.reshape(*args, **kwargs)
        return self

    def backward(self):
        for value in self.data.flatten():
            value.backward()

    def parameters(self):
        return list(self.data.flatten())

    def __repr__(self):
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def item(self):
        return self.data.flatten()[0].data
