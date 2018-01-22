
import numpy as np
import matplotlib.pyplot as plt
import json

iter = np.arange(0,40,1)
loss1 = json.load(open('CNN/loss.test.json'))[0:40]
acc1 = json.load(open('CNN/acc.test.json'))[0:40]
loss2 = json.load(open('MLP1/loss.test.json'))[0:40]
acc2 = json.load(open('MLP1/acc.test.json'))[0:40]
loss3 = json.load(open('MLP2/loss.test.json'))[0:40]
acc3 = json.load(open('MLP2/acc.test.json'))[0:40]
print max(acc1), max(acc2), max(acc3)
plt.figure()
plt.plot(iter, loss1, "g-",label="Loss: Convolutional(2 Layer)")
plt.plot(iter, acc1, "b-",label="Accuracy: Convolutional(2 Layer)")
plt.plot(iter, loss2, "r-",label="Loss: Linear(1 Layer)")
plt.plot(iter, acc2, "m-",label="Accuracy: Linear(1 Layer)")
plt.plot(iter, loss3, "y-",label="Loss: Linear(2 Layer)")
plt.plot(iter, acc3, "c-",label="Accuracy: Linear(2 Layer)")

#plt.axis([0.0,5.01,-1.0,1.5])
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.title("Loss/Accuracy-Epoch Graph")

plt.grid(True)
plt.legend()
plt.show()
