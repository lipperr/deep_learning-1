import numpy as np
import scipy.special as ss
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(np.zeros(input.shape), input)
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_relu = (input > 0).astype(int)
        return grad_output * grad_relu

        return super().compute_grad_input(input, grad_output)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return ss.expit(input)
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * ss.expit(input) * (1 - ss.expit(input))
        return super().compute_grad_input(input, grad_output)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        x = input - np.max(input, axis=1)[:, None]
        return ss.softmax(x, axis=1)
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        x = input - np.max(input, axis=1)[:, None]
        z = ss.softmax(x, axis=1)
        diag_z = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        diag = np.arange(z.shape[1])
        diag_z[:, diag, diag] = z
        dx = diag_z - np.einsum("ij,ik -> ijk", z, z)
        return np.einsum("ij,ijk->ik", grad_output, dx)

        return super().compute_grad_input(input, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        x = input - np.max(input, axis=1)[:, None]
        return ss.log_softmax(x, axis=1)
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        x = input - np.max(input, axis=1)[:, None]
        z = ss.softmax(x, axis=1)

        d = np.full((z.shape[0], z.shape[1], z.shape[1]), np.eye(z.shape[-1]))
        dx = d - np.einsum("ij,ik -> ijk", np.ones(z.shape), z)
        return np.einsum("ij,ijk->ik", grad_output, dx)

        return super().compute_grad_input(input, grad_output)
