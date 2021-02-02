import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer
from random import sample
#env = gym.make('FetchReach-v1')
class Actor(object):
    def __init__(self, sess, obs_dim, act_dim, action_bound, learning_rate, tau, batch_size):
        self.sess=sess
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.action_bound=action_bound
        self.learning_rate=learning_rate
        self.tau=tau
        self.batch_size=batch_size
        self.inputs,self.out,self.scaled_out=self.create_network_actor()
        self.network_params=tf.trainable_variables()
        self.inputs_target,self.out_target,self.scaled_out_target=self.create_network_actor()
        self.network_target_params=tf.trainable_variables()[len(self.network_params):]
        # weights
        self.update_network_target_params = \
            [self.network_target_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.network_target_params[i], 1. - self.tau))
                for i in range(len(self.network_target_params))]
        self.action_gradient=tf.placeholder(tf.float32,[None,self.act_dim])
        self.unnormal_gradients=tf.gradients(self.scaled_out,self.network_params,-self.action_gradient)
        self.actor_gradients=list(map(lambda x: tf.div(x, self.batch_size),self.unnormal_gradients))
        self.optimize= self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
        apply_gradients(zip(self.actor_gradients, self.network_params))
        self.num_trainable_vars ==len(self.network_params)+len(self.network_params)
    def create_network_actor(self):
        inputs=tflearn.input_data(shape=[None,self.obs_dim])
        out_1=tflearn.fully_connected(inputs,400)
        out_1=tflearn.layers.normalization.batch_normalization(out_1)
        out_1=tflearn.activations.relu(out_1)
        out_1=tflearn.fully_connected(out_1,300)
        out_1=tflearn.layers.normalization.batch_normalization(out_1)
        out_1=tflearn.activations.relu(out_1)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out_2= tflearn.fully_connected(out_1, self.act_dim, activation='tanh', weights_init=w_init)
        scaled_out=tf.multiply(out_2,self.action_bound)
    def train(self,inputs,act_gradient):
        return self.sess.run(self.optimize,feed_dict={
            self.inputs:inputs,
            self.action_gradient:act_gradient
            })
    def predict(self,inputs):
        return self.sess.run(self.scaled_out,feed_dict={
            self.inputs:inputs
            })
    def predict_target(self,inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
            })
    def update_target(self):
        return self.sess.run(self.update_network_target_params)
    def get_no_of_training_variables(self):
        return  self.num_trainable_vars
class Critic(object):
    def __init__(self, sess, obs_dim, act_dim, learning_rate, tau, gamma, num_actor_vector):
        self.sess=sess
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.learning_rate=learning_rate
        self.tau=tau
        self.gamma=gamma

        self.inputs,self.action,self.out=self.create_network_critic()
        self.network_params=self.trainable_variables()[num_actor_vector:]
        self.inputs_target,self.action_target,self.out_target=self.create_network_critic()
        self.network_target_params=self.trainable_variables()[(len(self.network_params)+num_actor_vector):]
        self.network_target_params = \
            [self.network_target_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.network_target_params[i], 1. - self.tau))
                for i in range(len(self.network_target_params))]
        self.target_q_value=tf.placeholder(tf.float32,[None,1])
        self.loss=tflearn.mean_square(self.target_q_value,self.out)
        self.optimize=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_gradient= tf.gradients(self.out, self.action)
    def create_network_critic(self):
        inputs=tflearn.input_data(shape=[None,self.obs_dim])
        action=tflearn.input_data(shape=[None,self.act_dim])
        out_1=tflearn.fully_connected(inputs,400)
        out_1=tflearn.layers.normalization.batch_normalization(out_1)
        out_1=tflearn.activations.relu(out_1)
        t1=tflearn.fully_connected(out_1,300)
        t2=tf_learn.fully_connected(action,300)
        out_1=tflearn.activation(tf.matmul(out_1,t1.W)+tf.matmul(action,t2.W)+ t2.b, activation='relu')
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(out_1, 1, weights_init=w_init)
        return inputs,action,out
    def train(self,inputs,action,target_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value:target_q_value
        })
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_gradient, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.network_target_params)
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
def con_st_goal(s):
    obs_1=np.reshape(s(['observation'],(1,10)))
    des_goal=np.reshape(s(['desired_goal'],(1,3)))
    return np.concatenate([obs_1,des_goal],axis=1)
def store_sample(s,a,r,d,info ,s2):
    ob_1 = np.reshape(s['observation'],(1,10))
    ac_1 = np.reshape(s['achieved_goal'],(1,3))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    ob_2 = np.reshape(s2['observation'],(1,10))
    ac_2=np.reshape(s['achieved_goal'],(1,3))
    de_2=np.reshape(s['desired_goal'],(1,3))
    s_1 = np.concatenate([ob_1,ac_1],axis=1)
    s2_1 = np.concatenate([ob_2,ac_1],axis=1)
    s_2 = np.concatenate([ob_1,de_1],axis=1)
    s2_2 =np.concatenate([ob_2,de_1],axis=1)
     
    substitute_reward = env.compute_reward(s['achieved_goal'],s['desired_goal'], info)

    replay_buffer.append((s_2,a,r,d,s2_2))
    
    replay_buffer.append((s_1,a,substitute_reward,True,s2_1))
def train(sess,env,actor,critic,actor_noise,buffer_size,random_seed,minibatch_size):
     sess.run(tf.global_variables_initializer())
     actor.update_target_network()
     critic.update_target_network()
     replay_buffer=ReplayBuffer(buffer_size,random_seed)
     minibatch_size=64
     for i in range(50000):
         s = env.reset()
         ep_reward=0
         
         for j in range(1000):
             R=0
             s_1=con_st_goal(s)
             a = actor.predict(np.reshape(s_1, (1, actor.obs_dim))) + actor_noise()
             s2,r, terminal, info = env.step(a[0])
             
             store_sample(s_1,a,r,terminal,info,s2)
             r_2=r
             R+=r_2
             
                 
             if replay_buffer.size() > int(minibatch_size):
                 s_batch_1, a_batch_1, r_batch_1, t_batch_1, s2_batch_1 = \
                    replay_buffer.sample_batch(minibatch_size)
                 s_batch = np.squeeze(np.array(s_batch_1),axis=1)
                 s2_batch = np.squeeze(np.array(s2_batch_1),axis=1)
                 r_batch=np.reshape(np.array(r_batch_1),(len(r_batch_1),1))
                 a_batch=np.array(a_batch_1)
                 d_batch=np.reshape(np.array(d_batch_1)+0,(64,1))
                 target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))
                 y_i = []
                 for k in range((minibatch_size)):
                     if t_batch[k]:
                         y_i.append(r_batch[k])
                     else:
                         y_i.append(r_batch[k] + critic.gamma * target_q[k])
                    #print(y_i)

               
                 predicted_q_value, _ = critic.train(
                     s_batch, a_batch, np.reshape(y_i,((minibatch_size), 1)))
                
                 ep_ave_max_q += np.amax(predicted_q_value)

               
                 a_outs = actor.predict(s_batch)
                 grads = critic.action_gradients(s_batch, a_outs)
                 actor.train(s_batch, grads[0])
                 actor.update_target_network()
                 critic.update_target_network()

             s = s2
             ep_reward += R
             if terminal:

                

                print(ep_reward,i)
                break
         
def main(random_seed,actor_lr,tau,critic_lr,gamma,minibatch_size):
     with tf.Session() as sess:
         env = gym.make('FetchReach-v1')
         np.random.seed(random_seed)
         tf.set_random_seed(random_seed)
         env.seed(random_seed)
         obs_dim =13
         act_dim =4
         action_bound = env.action_space.high
         assert (env.action_space.high == -env.action_space.low)
         actor = ActorNetwork(sess, obs_dim, act_dim, action_bound,
                             actor_lr,tau,minibatch_size)
         critic = CriticNetwork(sess, obs_dim, act_dim,critic_lr,tau,gamma,get_no_of_training_variables())
         actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_dim))
         buffer_size=1000000
         random_seed=1234
         batch_size=64
         train(sess, env, actor, critic, actor_noise,buffer_size,random_seed,minibatch_size)
if __name__ == '__main__':
    main(1234,.0001,.0001,.001,0.99,64)
         
