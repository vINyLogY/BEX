# coding: utf-8
r"""Backend for accelerated array-operations.
"""

from math import log10
from typing import Callable, Optional

import numpy as np
import torch
from torch.linalg import vector_norm as opt_vector_norm
from torch import sqrt as opt_sqrt
import torchdiffeq
from numpy.typing import ArrayLike, NDArray
# import opt_einsum as oe

# disable autograd by defalt
torch.set_grad_enabled(False)
DOUBLE_PRECISION = True
FORCE_CPU = True
ON_DEVICE_EIGEN_SOLVER = False
MAX_EINSUM_AXES = 52  # restrition from torch.einsum as of PyTorch 1.10
PI = np.pi


# Place to keep magic numbers
class Parameters:
    ode_rtol = 1.0e-5
    ode_atol = 1.0e-8
    svd_atol = 1.0e-8
    svd_ratio = 1

    if FORCE_CPU:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
        DOUBLE_PRECISION = False
    elif torch.backends.mps.is_available():
        device = 'mps'
        DOUBLE_PRECISION = False
    else:
        device = 'cpu'

    def __str__(self) -> str:
        string = f'ODE[{log10(self.ode_rtol):+.0f}]({log10(self.ode_atol):+.0f})'
        string += f' | SVD({log10(self.svd_atol):+.0f}, x{self.svd_ratio})'
        string += f' | Device:{self.device}'
        return string


parameters = Parameters()

# CPU settings
if DOUBLE_PRECISION:
    dtype = np.complex128
else:
    dtype = np.complex64
    parameters.ode_rtol = 1.0e-3
    parameters.ode_atol = 1.0e-6
    parameters.svd_atol = 1.0e-6
Array = NDArray[dtype]

moveaxis = np.moveaxis


def arange(n: int) -> Array:
    return np.arange(n, dtype=dtype)


def as_array(a: ArrayLike) -> Array:
    return np.array(a, dtype=dtype)


def zeros(shape: list[int]) -> Array:
    return np.zeros(shape, dtype=dtype)


def zeros_like(shape: list[int]) -> Array:
    return np.zeros_like(shape, dtype=dtype)


def ones(shape: list[int]) -> Array:
    return np.ones(shape, dtype=dtype)


def eye(m: int, n: int, k: int = 0) -> Array:
    return np.eye(m, n, k, dtype=dtype)


# GPU settings
if DOUBLE_PRECISION:
    opt_dtype = torch.complex128
else:
    opt_dtype = torch.complex64
OptArray = torch.Tensor


def opt_array(array: Array) -> OptArray:
    ans = torch.tensor(array, dtype=opt_dtype, device=parameters.device)
    return ans


def opt_sparse_array(array: Array) -> OptArray:
    ans = torch.tensor(array, dtype=opt_dtype, device=parameters.device)
    return ans.to_sparse_coo()


def opt_cat(tensors: list[OptArray]) -> OptArray:
    return torch.cat(tensors)


def opt_stack(tensors: list[OptArray]) -> OptArray:
    return torch.stack(tensors, dim=0)


def opt_eye_like(a: OptArray) -> OptArray:
    m, n = a.shape
    return torch.eye(m, n, device=parameters.device)


def opt_zeroes_like(a: OptArray):
    return torch.zeros_like(a)


def opt_split(tensors: OptArray, size_list: list[int]) -> list[OptArray]:
    return torch.split(tensors, size_list)


def opt_einsum(*args) -> OptArray:
    return torch.einsum(*args)


def opt_sum(array: OptArray, dim: int) -> OptArray:
    return torch.sum(array, dim=dim)


def opt_tensordot(a: OptArray, b: OptArray, axes: tuple[list[int],
                                                        list[int]]) -> OptArray:
    return torch.tensordot(a, b, dims=axes)


def opt_svd(a: OptArray,
            compressed=False) -> tuple[OptArray, OptArray, OptArray]:
    if not ON_DEVICE_EIGEN_SOLVER:
        a = a.cpu()
    # print(a.device)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    if not ON_DEVICE_EIGEN_SOLVER:
        u = u.to(device=parameters.device)
        s = s.to(device=parameters.device)
        vh = vh.to(device=parameters.device)

    if compressed:
        # Calculate rank from atol
        tol = parameters.svd_atol
        ratio = parameters.svd_ratio
        # default
        total_error = 0.0
        rank = 1
        for n, s_i in reversed(list(enumerate(s))):
            total_error += s_i
            if total_error > tol:
                rank = n + 1
                break
        # Enlarge by svd_ratio
        if ratio != 1:
            enlarged_rank = int(max(rank + 1, rank * ratio))
            rank = min(len(s), enlarged_rank)

        s = s[:rank]
        u = u[:, :rank]
        vh = vh[:rank, :]

    # ss = s.diag()
    return u, s, vh


def opt_qr(a: OptArray, compressed=False) -> tuple[OptArray, OptArray]:
    u, s, vh = opt_svd(a, compressed=compressed)
    return u, s.diag().to(opt_dtype) @ vh


def odeint(func: Callable[[float, OptArray], OptArray],
           y0: OptArray,
           dt: float,
           t0: float = 0.0,
           method='dopri5') -> tuple[OptArray, OptArray, int]:
    """Avaliable method:
    - Adaptive-step:
        - `dopri8` Runge-Kutta 7(8) of Dormand-Prince-Shampine
        - `dopri5` Runge-Kutta 4(5) of Dormand-Prince.
        - `bosh3` Runge-Kutta 2(3) of Bogacki-Shampine
        - `adaptive_heun` Runge-Kutta 1(2)
    - Fixed-step:
        - `euler` Euler method.
        - `midpoint` Midpoint method.
        - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
        - `explicit_adams` Explicit Adams.
        - `implicit_adams` Implicit Adams.
    - Scipy compatable (slow):
        - 'BDF'
    """

    t = torch.tensor([t0, t0 + dt], dtype=opt_dtype,
                     device=parameters.device).real
    if method == 'rk4':
        k1 = func(t0, y0) * dt
        k2 = func(t0 + dt / 3.0, y0 + k1 / 3.0) * dt
        k3 = func(t0 + dt * 2.0 / 3.0, y0 - k1 / 3.0 + k2) * dt
        k4 = func(t0 + dt, y0 + k1 - k2 + k3) * dt
        y1 = y0 + (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
    elif method == 'BDF':
        solution = torchdiffeq.odeint(func,
                                      y0,
                                      t,
                                      method='scipy_solver',
                                      options={'solver': 'BDF'})
        y1 = solution[1]
    else:
        solution = torchdiffeq.odeint(func,
                                      y0,
                                      t,
                                      method=method,
                                      rtol=parameters.ode_rtol,
                                      atol=parameters.ode_atol)
        y1 = solution[1]
    return y1


def opt_pinv(a: OptArray) -> OptArray:
    return torch.linalg.pinv(a, atol=parameters.svd_atol)


def opt_inv(a: OptArray) -> OptArray:
    return torch.linalg.inv(a)


def opt_unfold(tensor: OptArray, ax: int) -> OptArray:
    dim = tensor.shape[ax]
    ans = tensor.moveaxis(ax, 0).reshape((dim, -1))
    return ans


def opt_fold(vectors: OptArray, shape: list[int], ax: int):
    dim = shape[ax]
    _shape = [dim] + [n for i, n in enumerate(shape) if i != ax]
    assert dim == vectors.shape[0]
    ans = vectors.reshape(_shape).moveaxis(0, ax)
    return ans


def opt_transform(tensor: OptArray, ax: int, op: OptArray) -> OptArray:
    """Tensor-matrix contraction that keeps the indices convension of tensor.
    """
    shape = list(tensor.shape)
    ans_vectors = op @ opt_unfold(tensor, ax)
    return opt_fold(ans_vectors, shape, ax)


def opt_trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
    assert tensor1.shape == tensor2.shape
    assert 0 <= ax < tensor1.ndim
    vectors1 = opt_unfold(tensor1, ax)
    vectors2 = opt_unfold(tensor2, ax).transpose()
    return vectors1 @ vectors2
