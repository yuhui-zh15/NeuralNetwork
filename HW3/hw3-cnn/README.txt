#修改说明：
仅修改model.py
均在源代码框架基础上进行修改，已删除代码冗余部分(如输出训练数据，tensorboard记录等），详见下方详细修改说明
MNIST_data文件夹下为MNIST训练集数据

#使用说明：
python main.py即可运行

#作者：
张钰晖
计55
2015011372
Email: yuhui-zh15@mails.tsinghua.edu.cn
Tel: 185-3888-2881

#测试环境：
macOS 10.13

#代码详细修改说明：
- model.py

def __init__(...):
......
    logits = x
    logits = conv_layer(logits, 5, 1, 16)
    logits = batch_normalization_layer(logits, is_train)
    logits = relu_layer(logits)
    logits = max_pool_layer(logits, 2)
    logits = conv_layer(logits, 5, 16, 32)
    logits = batch_normalization_layer(logits, is_train)
    logits = relu_layer(logits)
    logits = max_pool_layer(logits, 2)
    logits = tf.reshape(logits, [-1, 1568])
    logits = linear_layer(logits, 1568, 10)
......

def batch_normalization_layer(inputs, isTrain=True):
    mean_total, variance_total = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False), tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False)
    mean, variance = tf.nn.moments(inputs, [0, 1, 2])
    decay = tf.constant(0.9)
    gamma = tf.Variable(tf.ones(inputs.get_shape()[-1]))
    beta = tf.Variable(tf.zeros(inputs.get_shape()[-1]))
    epsilon = tf.constant(1e-3)
    if isTrain:
        assign_mean = tf.assign(mean_total, mean_total * decay + mean * (1 - decay))
        assign_variance = tf.assign(variance_total, variance_total * decay + variance * (1 - decay))
        with tf.control_dependencies([assign_mean, assign_variance]):
            norm = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    else: norm = tf.nn.batch_normalization(inputs, mean_total, variance_total, beta, gamma, epsilon)
    return norm

def linear_layer(inputs, input_size, output_size, isTrain=True):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(inputs, W) + b

def conv_layer(inputs, kernel_size, channel_in, channel_out, isTrain=True):
    W = weight_variable([kernel_size, kernel_size, channel_in, channel_out])
    b = bias_variable([channel_out])
    return tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME') + b

def max_pool_layer(inputs, kernel_size, isTrain=True):
    return tf.nn.max_pool(inputs, ksize=[1, kernel_size, kernel_size, 1], strides=[1, kernel_size, kernel_size, 1], padding='SAME')

def relu_layer(inputs, isTrain=True):
    return tf.nn.relu(inputs)

def dropout_layer(inputs, keep_prob, isTrain=True):
    return tf.nn.dropout(inputs, keep_prob)