import numpy as np
import imgutil

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input
 
    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
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
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
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
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    n, c_in, h_in, w_in = input.shape
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    input_reshaped = input_padded.reshape(n, c_in, h_in / kernel_size, kernel_size, w_in / kernel_size, kernel_size)
    output = input_reshaped.mean(axis=3).mean(axis=4)

    # assert output.shape == (n, c_in, h_in / 2, w_in / 2)
    return output
            

def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    n, c_in, h_in, w_in = input.shape
    grad_input = grad_output.repeat(kernel_size, axis=2).repeat(kernel_size, axis=3) / (kernel_size * kernel_size)
    if pad > 0: grad_input = grad_input[:, :, pad:-pad, pad:-pad]

    # assert grad_input.shape == input.shape
    return grad_input
    