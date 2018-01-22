#修改说明：
修改main.py, model.py, cell.py
均在源代码框架基础上进行修改，详见下方详细修改说明

#使用说明：
python main.py即可运行笔者最终设计的模型：双向LSTM+单隐层MLP
其余模型均以注释形式存在model.py中，若要运行其他模型，去掉响应的注释即可

#作者：
张钰晖
计55
2015011372
Email: yuhui-zh15@mails.tsinghua.edu.cn
Tel: 185-3888-2881

#测试环境：
macOS 10.13.1

#代码详细修改说明：
- main.py
#todo: load word vector from 'vector.txt' to embed, where the value of each line is the word vector of the word in vocab_list
embed = []
embed_dict = {}
with open('%s/vector.txt' % (path)) as f:
    for line in f:
        splitline = line.strip().split(' ')
        # assert len(splitline) - 1 == FLAGS.embed_units
        word = splitline[0]
        vector = []
        for i in range(FLAGS.embed_units):
            vector.append(float(splitline[i + 1]))
        embed_dict[word] = vector
zero_vector = []
for i in range(FLAGS.embed_units):
    zero_vector.append(float(0))
for word in vocab_list:
    if word in embed_dict:
        embed.append(embed_dict[word])
    else:
        embed.append(zero_vector)
#todo: implement the tensorboard code recording the statistics of development and test set
loss, accuracy = evaluate(model, sess, data_dev)
print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
summary_dev = tf.Summary()
summary_dev.value.add(tag='loss/dev', simple_value=loss)
summary_dev.value.add(tag='accuracy/dev', simple_value=accuracy) 
summary_writer.add_summary(summary_dev, epoch)
loss, accuracy = evaluate(model, sess, data_test)
print("        test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
summary_test = tf.Summary()
summary_test.value.add(tag='loss/test', simple_value=loss)
summary_test.value.add(tag='accuracy/test', simple_value=accuracy) 
summary_writer.add_summary(summary_test, epoch)

- model.py
#todo: implement placeholders
self.texts = tf.placeholder(tf.string, shape=[None, None])  # shape: batch*len
self.texts_length = tf.placeholder(tf.int64, shape=[None])  # shape: batch
self.labels = tf.placeholder(tf.int64, shape=[None])  # shape: batch
'''
# BasicRNN Model
cell = BasicRNNCell(num_units)
outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
logits = tf.layers.dense(states, num_labels)
'''
'''
# BasicLSTM Model
cell = BasicLSTMCell(num_units)
outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
logits = tf.layers.dense(states[1], num_labels)
'''
'''
# GRU Model
cell = GRUCell(num_units)
outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
logits = tf.layers.dense(states, num_labels)
'''
#todo: implement unfinished networks
# BiLSTM Model
cell_fw = BasicLSTMCell(num_units)
cell_bw = BasicLSTMCell(num_units)
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
(cell_fw, hidden_fw), (cell_bw, hidden_bw) = states
logits = tf.layers.dense(tf.concat([hidden_fw, hidden_bw], axis=1), 256)
logits = tf.layers.dropout(logits)
logits = tf.layers.dropout(tf.layers.dense(logits, num_labels))

- cell.py
class BasicRNNCell(tf.contrib.rnn.RNNCell):
    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [batch_size, num_units]
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units])
            b = tf.get_variable('b', [self._num_units], initializer=tf.constant_initializer(0.0))    
            new_state = self._activation(tf.matmul(tf.concat([inputs, state], axis=1), W) + b)
        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [batch_size, num_units]
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            W_zr = tf.get_variable('W_zr', [FLAGS.embed_units + self._num_units, self._num_units * 2])
            b_zr = tf.get_variable('b_zr', [self._num_units * 2], initializer=tf.constant_initializer(1.0))
            z, r = tf.split(value=tf.sigmoid(tf.matmul(tf.concat([inputs, state], axis=1), W_zr) + b_zr), num_or_size_splits=2, axis=1)
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units])
            b = tf.get_variable('b', [self._num_units], initializer=tf.constant_initializer(0.0))
            h = self._activation(tf.matmul(tf.concat([inputs, r * state], axis=1), W) + b)
            new_h = z * state + (1 - z) * h
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, embed_units]
        # state: [[batch_size, num_units], [batch_size, num_units]]
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units * 4])
            b = tf.get_variable('b', [self._num_units * 4], initializer=tf.constant_initializer(0.0))
            f, i, o, c_tilde = tf.split(value=tf.matmul(tf.concat([inputs, h], axis=1), W) + b, num_or_size_splits=4, axis=1)
            new_c = tf.sigmoid(f + self._forget_bias) * c + tf.sigmoid(i) * self._activation(c_tilde)
            new_h = tf.sigmoid(o) * self._activation(new_c)
        return new_h, (new_c, new_h)
