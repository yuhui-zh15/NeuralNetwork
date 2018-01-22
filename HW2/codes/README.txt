# 修改说明：
- 修改了functions.py、loss.py、run_cnn.py三个文件
- 增加了imgutil.pyx、setup.py两个文件
- 详见下方详细修改说明
* graph文件夹下为绘制训练曲线的程序和Accuracy和Loss数据，其中CNN文件夹下为卷积神经网络数据，MLP1文件夹下为单层MLP数据，MLP2文件夹下为双层MLP数据


# 使用说明：
- 主程序：
* 先执行python setup.py build_ext --inplace编译cython文件
* 再执行python run_mlp.py即可训练
* 参数为双层CNN网络，准确率最高可达98.54%
- 绘图程序：
* 在graph文件夹下执行python graph.py即可，会绘出三种模型下的训练曲线

# 作者：
- 张钰晖
- 计55
- 2015011372
- Email: yuhui-zh15@mails.tsinghua.edu.cn
- Tel: 185-3888-2881

# 测试环境：
- macOS 10.13

# 详细修改：
- functions.py
```
import numpy as np
import imgutil

def conv2d_forward(input, W, b, kernel_size, pad):
    n, c_in, h_in, w_in = input.shape
    c_out, h_out, w_out = W.shape[0], h_in + 2 * pad - kernel_size + 1, w_in + 2 * pad - kernel_size + 1
    
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant') # [n, c_in, h_padded, w_padded]
    input_cols = imgutil.im2col(input_padded, kernel_size, n, c_in, h_in, w_in, h_out, w_out) # [c_in * kernel_size * kernel_size, h_out * w_out * n]

    W_cols = np.reshape(W, [c_out, c_in * kernel_size * kernel_size]) # [c_out, c_in * kernel_size * kernel_size]
    
    output = np.matmul(W_cols, input_cols) + np.reshape(b, [-1, 1]) # [c_out, h_out * w_out * n]
    output = np.reshape(output, [c_out, h_out, w_out, n]) # [c_out, h_out, w_out, n]
    output = np.transpose(output, [3, 0, 1, 2]) # [n, c_out, h_out, w_out]
    
    # assert output.shape == (n, c_out, h_out, w_out)
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    n, c_in, h_in, w_in = input.shape
    h_padded, w_padded = h_in + 2 * pad, w_in + 2 * pad
    c_out, h_out, w_out = W.shape[0], h_padded - kernel_size + 1, w_padded - kernel_size + 1
    
    grad_output_cols = np.reshape(np.transpose(grad_output, [1, 2, 3, 0]), [c_out, h_out * w_out * n]) # [c_out, h_out * w_out * n]
    W_cols = np.reshape(W, [c_out, c_in * kernel_size * kernel_size]) # [c_out, c_in * kernel_size * kernel_size]
    grad_input_cols = np.matmul(W_cols.T, grad_output_cols) # [c_in * kernel_size * kernel_size, h_out * w_out * n]
    grad_input = imgutil.col2im(grad_input_cols, kernel_size, n, c_in, h_padded, w_padded, h_out, w_out) # [n, c_in, h_padded, w_padded]
    if pad > 0: grad_input = grad_input[:, :, pad:-pad, pad:-pad] # [n, c_in, h_in, w_in]

    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant') # [n, c_in, h_padded, w_padded]
    input_cols = imgutil.im2col(input_padded, kernel_size, n, c_in, h_in, w_in, h_out, w_out) # [c_in * kernel_size * kernel_size, h_out * w_out * n]
    grad_W = np.matmul(grad_output_cols, input_cols.T) # [c_out, c_in * kernel_size * kernel_size]
    grad_W = np.reshape(grad_W, [c_out, c_in, kernel_size, kernel_size]) # [c_out, c_in, kernel_size, kernel_size]
    
    grad_b = np.sum(grad_output, axis=(0, 2, 3)) # [c_out]

    # assert grad_input.shape == input.shape
    # assert grad_W.shape == W.shape
    # assert grad_b.shape == b.shape
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    n, c_in, h_in, w_in = input.shape
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    input_reshaped = input_padded.reshape(n, c_in, h_in / kernel_size, kernel_size, w_in / kernel_size, kernel_size)
    output = input_reshaped.mean(axis=3).mean(axis=4)

    # assert output.shape == (n, c_in, h_in / 2, w_in / 2)
    return output
            

def avgpool2d_backward(input, grad_output, kernel_size, pad):
    n, c_in, h_in, w_in = input.shape
    grad_input = grad_output.repeat(kernel_size, axis=2).repeat(kernel_size, axis=3) / (kernel_size * kernel_size)
    if pad > 0: grad_input = grad_input[:, :, pad:-pad, pad:-pad]

    # assert grad_input.shape == input.shape
    return grad_input
```

- loss.py
```
class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        self.prob = (np.exp(input).T / np.exp(input).sum(axis=1)).T
        return - np.sum(target * np.log(self.prob)) / len(input)

    def backward(self, input, target):
        '''Your codes here'''
        return (self.prob - target) / len(input)
```

- run_cnn.py
```
model = Network()
model.add(Conv2D('conv1', 1, 4, 5, 2, 1))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 8, 3, 1, 1))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 392)))
model.add(Linear('fc3', 392, 10, 0.1))
```

- imgutil.pyx
```
import numpy as np
cimport numpy as np
cimport cython


def im2col(np.ndarray[np.float64_t, ndim=4] input, int kernel_size, int n, int c_in, int h_in, int w_in, int h_out, int w_out):
    cdef np.ndarray[np.float64_t, ndim=2] cols = np.zeros((c_in * kernel_size * kernel_size, h_out * w_out * n))
    cdef int i, j, t, u, p, q, r, c
    for i in range(h_out):
        for j in range(w_out):
            for t in range(n):
                r = i * w_out * n + j * n + t
                for u in range(c_in):
                    for p in range(kernel_size):
                        for q in range(kernel_size):
                            c = u * kernel_size * kernel_size + p * kernel_size + q
                            cols[c, r] = input[t, u, i + p, j + q]
    return cols


def col2im(np.ndarray[np.float64_t, ndim=2] cols, int kernel_size, int n, int c_in, int h_in, int w_in, int h_out, int w_out):
    cdef np.ndarray[np.float64_t, ndim=4] input = np.zeros((n, c_in, h_in, w_in))
    cdef int i, j, t, u, p, q, r, c
    for i in range(h_out):
        for j in range(w_out):
            for t in range(n):
                r = i * w_out * n + j * n + t
                for u in range(c_in):
                    for p in range(kernel_size):
                        for q in range(kernel_size):
                            c = u * kernel_size * kernel_size + p * kernel_size + q
                            input[t, u, i + p, j + q] += cols[c, r]
    return input
```

- setup.py
```
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy

extensions = [
  Extension('imgutil', ['imgutil.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)
```

