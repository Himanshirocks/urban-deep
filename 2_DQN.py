#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers
import matplotlib.pyplot as plt
import os
from collections import deque
import random

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.model = Sequential()
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.alpha = 0.0001

		self.model.add(Dense(self.action_size, input_dim=self.state_size,use_bias=True, activation='linear'))
		self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

	def save_model_weights(self, suffix):
	# Helper function to save your model / weights.
		pass

	def load_model(self, model_file):
	# Helper function to load an existing model.
		pass

	def load_model_weights(self,weight_file):
	# Helper funciton to load model weights. 
		pass

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		
		# env = QNetwork(environment_name)
		self.env = gym.make(environment_name)
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n

		self.replay_memory_size=50000
		self.memory = deque(maxlen=self.replay_memory_size)
		self.batch_size = 32
		self.burn_in = 10000

		if environment_name == 'MountainCar-v0':
			self.gamma = 1
			self.episodes = 3000 #given in handout
			self.iterations = 201 #check how many, stops at 200
			self.terminate = 0 #0 for this env
			self.environment_num = 1
			self.max_reward = -200
		elif environment_name == 'CartPole-v0':
			self.gamma = 0.99
			self.episodes = 10000000 #check how many
			self.iterations = 300 #given in handout
			self.terminate = 1
			self.environment_num = 2
			self.max_reward = 0
		self.train_epsilon = 0.85
		self.train_evaluate_epsilon = 0.05
		self.final_update = 50000


		self.q_network = QNetwork(environment_name)

	def epsilon_greedy_policy(self, q_values, train_test_var):
		# Creating epsilon greedy probabilities to sample from.             
		rand = np.random.uniform(0,1)
		#Different Episolns for fitting model and evaluating the training
		if train_test_var:
			if(rand<=self.train_epsilon):
				return np.random.randint(0,self.action_size)
			else:
				return self.greedy_policy(q_values)
		else:
			if(rand<=self.train_evaluate_epsilon):
				return np.random.randint(0,self.action_size)
			else:
				return self.greedy_policy(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values) 

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		save_dir = os.path.join(os.getcwd(), 'saved_models_lqn_replay')
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		os.chdir(save_dir)

		# total_t_iter = 0
		total_updates = 0
		success_count = 0

		starting_epsilon = self.train_epsilon


		for i_episode in range(self.episodes):

			if self.environment_num==2:
				if total_updates>1000000:
					break

			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])	
			# print(self.model.predict(state))

			print('Episode no ',i_episode+1)
			total_reward = 0
			ep_terminate = False

			print('Training epsilon is',self.train_epsilon)		

			for t_iter in range(self.iterations):
		
				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),1) 
				if total_updates<=200000:
					self.train_epsilon = -((starting_epsilon-0.05)/200000)*total_updates + starting_epsilon  #decay epsilon 0.5 to 0.05
				elif total_updates==0:
					self.train_epsilon = starting_epsilon;
				else:
					self.train_epsilon = 0.05


				# self.env.render()
				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(state,[1,self.state_size])	
				self.memory.append((state, action, reward, next_state, done))
	
				if len(self.memory) > self.burn_in:
					total_reward+=reward					
					minibatch = random.sample(self.memory, self.batch_size)
					for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
						m_q_value_prime = m_reward
						if not m_done:
							m_q_value_prime = m_reward + self.gamma * np.max(self.q_network.model.predict(m_next_state)[0])
							m_q_value_target = self.q_network.model.predict(m_state)
							m_q_value_target[0][m_action] = m_q_value_prime

							self.q_network.model.fit(m_state,m_q_value_target,epochs=1, verbose=0)			
							total_updates+=1	

				if (total_updates) % 10000 == 0:
					# print("Saving model at",total_updates)
					model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,total_updates) 
					filepath = os.path.join(save_dir, model_name)
					self.q_network.model.save(model_name)	

				if done:
					if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and t_iter+1>=200):
						print("success")
						success_count+=1

					if total_reward>self.max_reward:
						self.max_reward = total_reward

					print("Episode finished after {} iterations and %d success and max till now %d ".format(t_iter+1)%(success_count,self.max_reward)) 	
					break

				state=next_state

			# total_t_iter+=t_iter+1 
			print("Total updates is",total_updates)
			print('----------------')				


			
		print("Saving final model at",total_updates)
		model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,total_updates) 
		filepath = os.path.join(save_dir, model_name)
		self.q_network.model.save(model_name)
		print("Total success is %d and highest is %d " %(success_count,self.max_reward))

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		num_episodes = 20
		plot_mat = np.zeros((num_episodes, (self.final_update/10000) + 1))
		load_dir = os.path.join(os.getcwd(), 'saved_models_lqn_final')
		print(self.final_update)
		for i in range(10000, self.final_update + 10000, 10000):
			model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,i)
			filepath = os.path.join(load_dir, model_name)
			model = load_model(filepath)
			for e in range(num_episodes):
				total_epi_reward = 0
				print('Episode no ',e+1)
				state = self.env.reset()
				state = np.reshape(state,[1,self.state_size])	

				for t_iter in range(self.iterations):

					action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0) 					

					self.env.render()
					next_state, reward, done, info = self.env.step(action)
					next_state = np.reshape(state,[1,self.state_size])	
					total_epi_reward+=reward
					
					# if (self.terminate==0 and state[0][0]==0.5) or (self.terminate==1 and t_iter==200):
					# 	print("success")
					# 	ep_terminate = True
					# 	break

					if done:
						print("Model %d" %(i))
						print("Episode finished after {} iterations with episodic reward %d".format(t_iter+1)%(total_epi_reward)) 
						plot_mat[e,(i/10000) - 1] = total_epi_reward
						break			
					state = next_state

		x = range(1,num_episodes + 1)
		for i in range(len(x)):
			plot_mat[i,(self.final_update/10000)] = x[i]


		for i in range(1,self.final_update/10000):
			plt.figure(i)
			print(plot_mat[:,(self.final_update/10000)])
			print(plot_mat[:,i])
			plt.plot(plot_mat[:,(self.final_update/10000)], plot_mat[:,i])
			plt.title('%d Model' %(i))
			# plt.show()			
			plt.savefig('graph_%d.eps' %(i),bbox_inches='tight' )

		exit()

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	dqn_agen = DQN_Agent(environment_name)

	dqn_agen.train()

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

