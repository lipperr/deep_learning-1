import numpy as np
import scipy.stats as st
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        unbiased_y = input @ self.weight.T 
        return unbiased_y + self.bias if self.bias is not None else unbiased_y
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """ 
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        # dl/dx = dl/df * df/dx
        return grad_output @ self.weight 
        return super().compute_grad_input(input, grad_output)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """ 
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        # dl/dw = dl/df * df/dw
        self.grad_weight +=  grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)
        super().update_grad_parameters(input, grad_output)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store these values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        self.mean = input.mean(axis=0)
        self.input_mean = input - self.mean[None, :]
        self.var = np.var(input, axis=0)
        self.sqrt_var = np.sqrt(self.var + self.eps)
        self.inv_sqrt_var = 1 / self.sqrt_var
        
        if self.training: # train mode
            self.norm_input = self.input_mean * self.inv_sqrt_var[None, :]
            B = input.shape[1]
            self.running_mean =  (1-self.momentum) * self.running_mean + self.momentum * self.mean 
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * self.var * B / (B - 1)

        else: # eval mode
            self.norm_input = (input - self.running_mean[None, :]) / np.sqrt(self.running_var + self.eps)[None, :]
             
            
        if self.affine:
            return self.weight[None, :] * self.norm_input + self.bias

        return self.norm_input
        
        
    def compute_grad_input(self, input: np.ndarray,  grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        (B, NF) = input.shape
        dxhat = grad_output
        if self.affine:
            dxhat = grad_output * self.weight
            
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        if self.training:
            dxmu2 =  self.input_mean * 1. /np.sqrt(self.var+self.eps) * (-1. /(self.sqrt_var**2) * np.sum(dxhat*self.input_mean, axis=0)) / B
            dx1 = (dxhat * self.inv_sqrt_var + dxmu2)
            dmu = -1 * np.sum(dxhat * self.inv_sqrt_var+dxmu2, axis=0)
            dx2 = np.ones(input.shape) * dmu / B
            dx = dx1 + dx2
            return dx
            
        else:
            return dxhat / np.sqrt(self.running_var + self.eps)

    

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)


    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask =  st.bernoulli.rvs(1 - self.p, size=input.shape)
            return 1/(1-self.p) * self.mask * input
        return input
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            return 1/(1-self.p) * self.mask * grad_output
        return grad_output
        return super().compute_grad_input(input, grad_output)

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        for module in self.modules:
            input = module.forward(input)
            
        return input
        

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        if self.training:
            self.modules.reverse()
            for i, module in enumerate(self.modules[:-1]):
                grad_output = module.backward(self.modules[i+1].output, grad_output)
            self.modules.reverse()
            return self.modules[0].backward(input, grad_output)
        

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
