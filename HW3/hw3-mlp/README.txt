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
    logits = self.x_ # Input
    logits = linear_layer(logits, 784, 400) # Linear
    logits = batch_normalization_layer(logits, is_train) # BN
    logits = relu_layer(logits) # Relu
    logits = linear_layer(logits, 400, 10) # Linear
......

def batch_normalization_layer(inputs, isTrain=True):
    mean_total, variance_total = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False), tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False)
    mean, variance = tf.nn.moments(inputs, [0])
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

def relu_layer(inputs, isTrain=True):
    return tf.nn.relu(inputs)

def dropout_layer(inputs, keep_prob, isTrain=True):
    return tf.nn.dropout(inputs, keep_prob)