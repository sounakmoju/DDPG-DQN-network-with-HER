import numpy as np 
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import Input,Flatten,Dense,Embedding,Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
import gym
from collections import deque
import random
class Replay_buffer(object):
    def __init__(self, mem_size, random_seed=123):
        self.mem_size=mem_size
        self.mem=deque()
        random.seed(random_seed)
        self.leng=0
    
    def store_transition(self,obs,act,rwd,obs_1,done):
        experience_1 =(obs,act,rwd,obs_1,done)
        if self.leng < self.mem_size:
            self.mem.append(experience_1)
            self.leng+=1
        else:
            self.mem.popleft()
            self.mem.append(self,obs,act,rwd,obs_1,done)
    def size(self):
        return self.leng
    
    def sample(self, batch_size):
        
        batch=random.sample(self.mem,batch_size)
        
        return batch
class Env(object):
    
    
    def __init__(self, bits):
        self.bits = bits
        self.obs=np.zeros(15)
    def goal(self):
        self.goal=np.random.randint(0, 2, size=self.bits)
        return self.goal
    def reset(self):
        #self.obs=np.random.randint(0, 2, size=self.bits)
        #self.goal=np.random.randint(0, 2, size=self.bits)
        #while (self.obs!=self.goal).all():
        self.obs=np.random.randint(0, 2, size=self.bits)
        return self.obs
    def step(self,action):
        self.obs=np.copy(self.obs)
        self.obs[action]=not self.obs[action]
        reward=-1
        done=False
        if (self.obs==self.goal).all():
            done=True
            reward=0.0
        return np.copy(self.obs),reward,done
class DQN(object):
    def __init__(self,obs_dim,act_dim,gamma,epsilon):
        optimizer = Adam(learning_rate=0.01)
        self.optimizer=optimizer
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.gamma=gamma
        self.epsilon=epsilon
        self.q_network=self.build_network()
        self.target_network=self.build_network()
        self.update_target_model()
        self.memory= Replay_buffer(1000000,1234)
    def build_network(self):
        model=Sequential()
        #model.add(Flatten(self.obs_dim,))
        model.add(Reshape((self.obs_dim,)))
        print(self.obs_dim)
        #model.add(Flatten(input_shape=(1,) + self.obs_dim))
        model.add(Dense(50,activation='relu',kernel_initializer='random_normal',bias_initializer='zeros'))
        model.add(Dense(self.act_dim,activation='linear',kernel_initializer='random_normal',bias_initializer='zeros'))
        model.compile(loss='mse',optimizer=self.optimizer)
        x = tf.random.normal((1,30))
        print((self.obs_dim))
        output = model(x)
        
        print(model.weights)
        return model
    def update_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    def act(self,observation):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0,self.act_dim)

        else:
            q_values = self.q_network.predict(observation,steps=1)
            #print(q_values[0])
            return np.argmax(q_values[0])
    def learn(self,b_size):
        minibatch=self.memory.sample(b_size)
        for observation,action,reward,next_observation,termination in minibatch:
            target=self.q_network.predict(observation,steps=1)
            if termination:
                target[0][action]=reward
            else:
                t=self.target_network.predict(next_observation,steps=1)
                target[0][action]=reward+self.gamma*np.amax(t)
            self.q_network.fit(observation,target,epochs=1,steps_per_epoch=1)
bit_size = 15
env = Env(bit_size)
agent=DQN(obs_dim=bit_size*2,act_dim=bit_size,gamma=0.95,epsilon=0.1)
episode=5000
epi_len=500
HER=True
future_k=4
b_size=64
agent.q_network.summary()
goal=env.goal()

for i in range(episode):
    ep_rwd=0
    obs_0=env.reset()
    ##print(obs_0.shape)
    
    epi_experience=[]
    for j in range(epi_len):
        conc_obs=tf.convert_to_tensor(np.concatenate([obs_0,goal], axis =-1), dtype=tf.float32)
        #conc_obs_12=np.concatenate([obs_0,goal], axis =-1)
        # print((conc_obs_12.shape))
        act=agent.act(conc_obs)
        obs_1,rwd,done=env.step(act)
        epi_experience.append((obs_0, act, rwd, obs_1, goal, done))
        
        obs_0=obs_1
        ep_rwd+=rwd
        if done:
            break
        
        print(ep_rwd,i)
    for j in range(epi_len):
        obs_0, act, rwd, obs_1, goal, done = epi_experience[j]
        #inputs0=(np.concatenate([obs_0, goal], axis=-1)).resize((30,))
        #inputs1=(np.concatenate([obs_1, goal], axis=-1)).resize((30,))
        
                
        inputs0 = tf.convert_to_tensor(np.concatenate([obs_0, goal], axis=-1),dtype=tf.float32)
        inputs1 = tf.convert_to_tensor(np.concatenate([obs_1, goal], axis=-1),dtype=tf.float32)
        #tf.print((inputs0))
        agent.memory.store_transition(inputs0,act,rwd,inputs1,done)
        if HER:
            for h in range(future_k):
                future_goal = np.random.randint(j,epi_len)
                goal_ach = epi_experience[future_goal][3]
                #new_inputs0=(np.concatenate([obs_0, goal_], axis=-1)).resize((30,))
                #new_inputs1=(np.concatenate([obs_1, goal_], axis=-1)).resize((30,))
                
                new_inputs0 = tf.convert_to_tensor(np.concatenate([obs_0, goal_ach],axis=-1),dtype=tf.float32)
                #n=(np.concatenate([obs_0, goal_],axis=-1))
                #n_1=tf.convert_to_tensor(n)
                #print(n)
                
                
                new_inputs1 = tf.convert_to_tensor(np.concatenate([obs_1, goal_ach], axis=-1),dtype=tf.float32)
                #tf.print((n_1))
                if (np.array(obs_1) == np.array(goal_ach)).all():
                    r_ = 0.0
                    done=True
                    
                else:
                    r_ = -1.0
            agent.memory.store_transition(new_inputs0, act, r_, new_inputs1, done)
        
    
    j=0
    while(j<1):
        if agent.memory.size() > b_size:
            agent.learn(b_size)
            agent.update_target_model()
            j=+1
            
        
                
        
            

                
