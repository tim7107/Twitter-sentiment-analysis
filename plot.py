import numpy as np
import matplotlib.pyplot as plt
lstm = np.array([76.67,77.32,77.61,77.78,77.92,77.97,78.07,78.11,78.17,78.26])
rnn = np.array([78.06,78.16,77.99,77.80,77.29,76.47,76.12,76.00,75.37,75.22])
print(lstm)
print(type(lstm))
bert_case1 = np.array([78.2,78.61,78])
bert_case2 = np.array([84.24,84.92,84.82])
bert_case3 = np.array([85.49,86.08,85.89])
Naive_Bayes = np.array([77.01])
LR = np.array([77.63])
DT = np.array([71.36])


plt.plot(range(10),lstm,color='black',label='LSTM')
plt.plot(range(3),bert_case1,color='red',label='bert_case1')
plt.plot(range(3),bert_case2,color='blue',label='bert_case2')
plt.plot(range(3),bert_case3,color='green',label='bert_case3')
plt.plot(range(10),rnn,color='orange',label='rnn')
plt.plot(range(1),Naive_Bayes,'rs',label='Naive_Bayes')
plt.plot(range(1),LR,'^k',label='LR')
plt.plot(range(1),DT,'go-',label='DT')
plt.legend()
plt.title("Result Comparsion")
plt.ylabel("Training Accuracy") # y label
plt.xlabel("Epoch") # x label
plt.savefig("result.png")
