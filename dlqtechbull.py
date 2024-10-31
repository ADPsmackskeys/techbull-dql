import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
import math
from numpy.random import choice
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Model Packages for reinforcement learning
from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

print ("ooh ooh")
dataset = read_csv ('data/tcs.csv', index_col = 0)

#Disable the warnings
import warnings
warnings.filterwarnings ('ignore')

set_option ('display.precision', 3)
dataset.describe ()
dataset['Close'].plot ()

X = list (dataset["Close"])
X = [float (x) for x in X]
train_size = int (len (X) * 0.8)
X_train, X_test = X[0 : train_size], X[train_size : len (X)]

class Agent:
    def __init__ (self, state_size, is_eval = False, model_name = ""):
        #State size depends and is equal to the the window size, n previous days
        self.state_size = state_size # normalized previous days, 
        self.action_size = 3 # hold, buy, sell
        self.memory = deque (maxlen = 1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model (model_name) if is_eval else self._model ()

    def _model (self):
        model = Sequential ()
        #Input Layer
        model.add (Dense (units = 64, input_dim = self.state_size, activation = "relu"))
        #Hidden Layers
        model.add (Dense (units = 32, activation = "relu"))
        model.add (Dense (units = 8, activation = "relu"))
        #Output Layer 
        model.add (Dense (self.action_size, activation = "linear"))
        model.compile (loss = "mse", optimizer = Adam (learning_rate = 0.001))
        return model
    
    def act (self, state): 
        # If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        # actions suggested.
        if not self.is_eval and random.random () <= self.epsilon:
            return random.randrange (self.action_size)        
        options = self.model.predict (state)
        
        # action is based on the action that has the highest value from the q-value function.
        return np.argmax (options[0])

    def expReplay (self, batch_size):
        mini_batch = []
        l = len (self.memory)
        for i in range (l - batch_size + 1, l):
            mini_batch.append (self.memory[i])
        
        # the memory during the training phase. 
        for state, action, reward, next_state, done in mini_batch:
            target = reward # reward or Q at time t    
            # update the Q table based on Q table equation
            # set_trace()
            if not done:
                # set_trace()
                # max of the array of the predicted. 
                target = reward + self.gamma * np.amax (self.model.predict (next_state) [0])     
                
            # Q-value of the state currently from the table    
            target_f = self.model.predict (state)
            # Update the output Q table for the given action in the table     
            target_f[0][action] = target
            # train and fit the model where state is X and target_f is Y, where the target is updated. 
            self.model.fit (state, target_f, epochs = 1, verbose = 0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# prints formatted price
def formatPrice (n):
    return ("-₹" if n < 0 else "₹") + "{0:.2f}".format (abs (n))

def sigmoid (x):
    return 1 / (1 + math.exp (-x))

# returns an an n-day state representation ending at time t

def getState (data, t, n):    
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1] # pad with t0
    
    res = []
    for i in range (n - 1):
        res.append (sigmoid (block[i + 1] - block[i]))
    return np.array ([res])

# Plots the behavior of the output
def plot_behavior (data_input, states_buy, states_sell, profit, e):
    fig = plt.figure (figsize = (15, 5))
    plt.plot (data_input, color = 'k', lw = 2.)
    plt.plot (data_input, '^', markersize = 10, color = 'g', label = 'Buying signal', markevery = states_buy)
    plt.plot (data_input, 'v', markersize = 10, color = 'r', label = 'Selling signal', markevery = states_sell)
    plt.title ('Total gains: %f' % (profit))
    plt.legend()
    plt.savefig('output/' + "episode_" + str (e) + '.png')
    # plt.show()

window_size = 1
agent = Agent (window_size)
#In this step we feed the closing value of the stock price 
data = X_train
l = len (data) - 1
batch_size = 32
#An episode represents a complete pass over the data.
episode_count = 10

for e in range (episode_count + 1):
    print("Running episode " + str (e) + "/" + str (episode_count))
    state = getState (data, 0, window_size + 1)
    #set_trace()
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    for t in range (l):
        action = agent.act (state)    
        # hold
        next_state = getState (data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append (data[t])
            states_buy.append (t)
            #print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len (agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)      
            reward = max (data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            states_sell.append (t)
            #print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        #appends the details of the state action etc in the memory, which is used further by the exeReply function
        agent.memory.append ((state, action, reward, next_state, done))
        state = next_state

        if done:
            print ("--------------------------------")
            print ("Total Profit: " + formatPrice(total_profit))
            print ("--------------------------------")
            #set_trace()
            #pd.DataFrame(np.array(agent.memory)).to_csv("Agent"+str(e)+".csv")
            #Chart to show how the model performs with the stock goin up and down for each 
            plot_behavior (data, states_buy, states_sell, total_profit, e)

        if len (agent.memory) > batch_size:
            agent.expReplay (batch_size)    
            

    agent.model.save ("model_ep" + str(e) + ".keras")

#Deep Q-Learning Model
print (agent.model.summary())

test_data = X_test
l_test = len (test_data) - 1
state = getState (test_data, 0, window_size + 1)
total_profit = 0
is_eval = True
done = False
states_sell_test = []
states_buy_test = []
#Get the trained model
model_name = "model_ep" + str (episode_count) + ".keras"
agent = Agent (window_size, is_eval, model_name)
state = getState (data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range (l_test):
    action = agent.act (state)
    #print(action)
    #set_trace()
    next_state = getState (test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1:
        agent.inventory.append (test_data[t])
        states_buy_test.append (t)
        print ("Buy: " + formatPrice(test_data[t]))

    elif action == 2 and len (agent.inventory) > 0:
        bought_price = agent.inventory.pop (0)
        reward = max (test_data[t] - bought_price, 0)
        #reward = test_data[t] - bought_price
        total_profit += test_data[t] - bought_price
        states_sell_test.append (t)
        print ("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))

    if t == l_test - 1:
        done = True
    agent.memory.append ((state, action, reward, next_state, done))
    state = next_state

    if done:
        print ("------------------------------------------")
        print ("Total Profit: " + formatPrice (total_profit))
        print ("------------------------------------------")
        
plot_behavior (test_data, states_buy_test, states_sell_test, total_profit, 11)