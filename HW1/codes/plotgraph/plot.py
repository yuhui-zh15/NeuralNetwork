import numpy as np
import matplotlib.pyplot as plt
import json

iter = np.arange(0,101,1)
loss = np.array(json.load(open('loss.json')))
acc = np.array(json.load(open('acc.json')))
plt.figure()
plt.plot(iter, loss, "g-",label="loss")
plt.plot(iter, acc, "b-",label="accuracy")

#plt.axis([0.0,5.01,-1.0,1.5])
plt.xlabel("epoch")
plt.ylabel("loss/accuracy")
plt.title("loss/accuracy-epoch graph")

plt.grid(True)
plt.legend()
plt.show()
