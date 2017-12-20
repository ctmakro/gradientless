import numpy as np

inp = 4.5 # input value (fixed)
p = 0 # parameter to learn
target = 9 # network should output

def network(inp): # network function
    return inp*p

def loss(result): # loss function
    return (result-target)**2 # MSE

mean_p = p # initial p
mean_l = loss(network(inp)) # initial loss

alpha = 0.05 # learning rate
beta = 0.03 # mean loss update rate
gamma = 0.1 # exploration rate

def update_mean_l(l):
    global mean_l
    mean_l = mean_l*beta + l *(1-beta)

def update_all(): # one iteration
    global p,mean_p,mean_l
    p = mean_p + np.random.normal() * gamma # randomly shift the weights.
    l = loss(network(inp)) # network forward

    dp = p - mean_p # delta weight
    dl = l - mean_l # delta loss

    mean_p = mean_p - alpha * dp * dl # gradient = dp * dl
    mean_l = mean_l * (1-beta) + l * beta

import matplotlib
from matplotlib import pyplot as plt

plog=[]
llog=[]
for i in range(250):
    update_all()
    print('iter',i)
    print(mean_p)
    print(mean_l)
    plog.append(mean_p)
    llog.append(mean_l)

plt.plot(plog,label='mean_p')
plt.plot(llog,label='mean_l')
plt.legend()
plt.show()
