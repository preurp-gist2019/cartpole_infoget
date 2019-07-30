# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import cartpole_mod
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import json
# Version 1.3

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.num_node = 24
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.train_info = []        # for Training accuracy & loss

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.num_node, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_node, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate),
                     metrics=['accuracy'])
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        hist_batch = [] # for training accuracy & loss
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            hist = self.model.fit(state, target, epochs=1, verbose=0)
            # hist has the length of batch size
            # Mean accuracy & loss should be calculated
            hist_batch.append([hist.history['acc'], hist.history['loss']])
            # self.model.fit(state, target, epochs=1, verbose=0)                 #######필요한지 확인바람
        
        hist_batch = np.array(hist_batch)
        acc_loss = [np.mean(hist_batch[:,0]), np.mean(hist_batch[:,1])]
        self.train_info.append(acc_loss)    # Save the training acc & loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_history(self):       
        data = self.train_info
        self.train_info = []  # For new episode, it should be cleared
        return data
            
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))
    env = gym.make('CartPole-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32
    
    
    history = {}
    history['score'] = []
    history['epsilon'] = []
    history['mean'] = []
    latest_hundred = 0
    temp_score = 0
    temp_epsilon = 0

    
    records = np.zeros((1,2))
    
    trialnumber = "Default_T4"#.format(agent.epsilon) #이부분을 자기가 바꿀것!
    
    #Make the bestscore file into 1
    #i = 1
    #f = open("./save/cartpole_ddqn_Bestscore_{0}.txt".format(trialnumber), 'w')
    #f.write("{}" .format(i))
    #f.close()
    
    #If bestscore is bigger than 50, epsilon must be 0.1. Only one time
    #f = open("./save/cartpole_ddqn_Bestscore.txt", 'r')
    #exscore = float(f.readline()) 
    #f.close()
    #if exscore >= 50 :
    #    agent.epsilon = 0.1
    
    for e in range(EPISODES):
        #agent.load("./save/cartpole-ddqn.h5")
        #done = False
        batch_size = 32
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        temp_epsilon = 0
        tmep_score = 0
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                temp_epsilon = agent.epsilon
                temp_score = time
                if e == 0:
                    records[0] = np.array([e, time])
                else:
                    records = np.concatenate((records, [[e, time]]), axis=0)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        if done is False :
            
            temp_score = 499
            temp_epsilon = agent.epsilon
            agent.update_target_model()
            records = np.concatenate((records, [[e, temp_score]]), axis=0)
            print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, temp_score, agent.epsilon))
        #Save the ex-bestscore to continue next time        
        if records[-1,1] >= np.amax(records[:,1]):
            #f = open("./save/cartpole_ddqn_Bestscore_{0}.txt".format(trialnumber), 'r')
            #line = float(f.readline())
            #f.close()
            #if records[-1,1] >= line :
            #    f = open("./save/cartpole_ddqn_Bestscore_{0}.txt".format(trialnumber), 'w')
            #    f.write("{}" .format(records[-1,1]))
            #    f.close()
            agent.save("./save/cartpole-ddqn_{0}.h5".format(trialnumber))
            print("Best model saved - episode: {}, score: {}".format(records[-1,0], records[-1,1]))
        history[e] = agent.get_history()
        history['score'].append(temp_score)
        history['epsilon'].append(temp_epsilon)
        
        if e >= 100:
            latest_hundred = np.mean(history['score'][-100:])
            print("Mean Score of previous 100 episodes: ", latest_hundred)
            
            if latest_hundred >= 195:
                print("Training Done!!! Mean Score: ", latest_hundred)
                break
        history['mean'].append(latest_hundred)
        
        hist_json = json.dumps(history)
        f = open("./logs/history_{0}.json".format(trialnumber), 'w')
        f.write(hist_json)
            
    history['mean'].append(latest_hundred)
    hist_json = json.dumps(history)
    f = open("./logs/history_{0}.json".format(trialnumber), 'w')
    f.write(hist_json)
    f.close()
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
