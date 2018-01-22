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
