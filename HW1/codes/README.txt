#修改说明：
仅修改了loss.py、layers.py、run_mlp.py三个文件
均在源代码框架基础上进行修改，无其它冗余代码部分，详见下方详细修改说明
data文件夹下为MNIST训练集数据
raw_data文件夹下提供了11个模型训练所得的Accuracy数据、Loss数据、图表以及绘图程序，如果感兴趣希望查看实验全部结果，详见该文件夹下README

#使用说明：
主程序：python run_mlp.py即可运行，参数为双隐层Relu激活网络，准确率最高可达98.69%
绘图程序：如果感兴趣希望使用，详见该文件夹下README

#作者：
张钰晖
计55
2015011372
Email: yuhui-zh15@mails.tsinghua.edu.cn
Tel: 185-3888-2881

#测试环境：
macOS 10.13

#代码详细修改说明：
- loss.py
class EuclideanLoss(object):
    def forward(self, input, target):
        return 0.5 * np.sum((target - input) ** 2) / len(input)
    def backward(self, input, target):
        return (input - target) / len(input)
class SoftmaxCrossEntropyLoss(object):
    def forward(self, input, target):
        self.prob = (np.exp(input).T / np.exp(input).sum(axis=1)).T
        return - np.sum(target * np.log(self.prob)) / len(input)
    def backward(self, input, target):
        return (self.prob - target) / len(input)

- layers.py
class Relu(Layer):
    def forward(self, input):
        self._saved_for_backward(input)
        return np.maximum(input, 0)
    def backward(self, grad_output):
        return grad_output * np.array(self._saved_tensor > 0, dtype = float)
class Sigmoid(Layer):
    def forward(self, input):
        self._saved_for_backward(1 / (1 + np.exp(-input)))
        return self._saved_tensor
    def backward(self, grad_output):
        return grad_output * self._saved_tensor * (1 - self._saved_tensor)
class Linear(Layer):
    def forward(self, input):
        self._saved_for_backward(input)
        return np.dot(input, self.W) + self.b
    def backward(self, grad_output):
        self.grad_W = np.dot(self._saved_tensor.T, grad_output)
        self.grad_b = grad_output
        return np.dot(grad_output, self.W.T)

- run_mlp.py
model = Network()
model.add(Linear('fc1', 784, 400, 0.01))
model.add(Relu('relu1'))
model.add(Linear('fc2', 400, 200, 0.01))
model.add(Relu('relu2'))
model.add(Linear('fc3', 200, 10, 0.01))
loss = EuclideanLoss(name='loss')
config = {
    'learning_rate': 1e-1,
    'weight_decay': 1e-4,
    'momentum': 1e-4,
    'batch_size': 100,
    'max_epoch': 200,
    'disp_freq': 100,
    'test_epoch': 1
}

