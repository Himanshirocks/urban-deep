#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers
import matplotlib.pyplot as plt
import os

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

		if environment_name == 'MountainCar-v0':
			self.gamma = 1
			self.episodes = 3000 #given in handout
			self.iterations = 201 #check how many, stops at 200
			self.terminate = 0 #0 for this env
			self.environment_num = 1
		elif environment_name == 'CartPole-v0':
			self.gamma = 0.99
			self.iterations = 1000000 #given in handout
			self.episodes = 5000 #check how many
			self.terminate = 1
			self.environment_num = 2
		self.train_epsilon = 0.5
		self.train_evaluate_epsilon = 0.05
		self.final_update = 50000
		# episodes = 100
		#the network stuff comes here
		self.q_network = QNetwork(environment_name)


	def epsilon_greedy_policy(self, q_values, train_test_var):
		# Creating epsilon greedy probabilities to sample from.             
		rand = np.random.uniform(0,1)
		#Different Episolns for fitting model and evaluating the training
		if train_test_var:
			if rand<=self.train_epsilon:
				return np.random.randint(0,self.action_size)
			else:
				return self.greedy_policy(q_values)
		else:
			if rand<=self.train_evaluate_epsilon:
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

		save_dir = os.path.join(os.getcwd(), 'saved_models_lqn_final')
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		os.chdir(save_dir)

		# total_t_iter = 0
		total_updates = 0
		success_count = 0

		starting_epsilon = self.train_epsilon

		for i_episode in range(self.episodes):
			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])	
			# print(self.model.predict(state))
			print('Episode no ',i_episode+1)
			total_reward = 0
			ep_terminate = False

			print('training epsilon is',self.train_epsilon)		

			for t_iter in range(self.iterations):

				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),1) 
				
				if self.environment_num == 1:
					if total_updates<=100000:
						self.train_epsilon = -((starting_epsilon-0.05)/100000)*total_updates + starting_epsilon  #decay epsilon 0.5 to 0.05
					else:
						self.train_epsilon = 0.05
				elif self.environment_num==2:
					if total_updates<=30000:
						self.train_epsilon = -((starting_epsilon-0.05)/30000)*total_updates + starting_epsilon
					else:
						self.train_epsilon = 0.05

				# if total_updates<=100000:#TO BE CHANGED
				# 	self.train_epsilon = self.train_epsilon - 0.45e-05 #decay epsilon

				# if i_episode >=2500:
				# 	self.env.render()
				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(state,[1,self.state_size])	
				total_reward+=reward
				# if total_updates==100:

				if done:
					q_value_prime = reward
					q_value_target = self.q_network.model.predict(state)
					q_value_target[0][action] = q_value_prime
					self.q_network.model.fit(state,q_value_target,batch_size=None,epochs=1, verbose=0) #doing sgd
					if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and t_iter+1==200):
						success_count+=1
						print("success")
						# ep_terminate = True
					print("Episode finished after {} iterations with %d rewards and %d success".format(t_iter+1)%(total_reward,success_count)) 
					break
				else:
					q_value_prime = reward + self.gamma * np.max(self.q_network.model.predict(next_state)[0])

				q_value_target = self.q_network.model.predict(state)
				q_value_target[0][action] = q_value_prime
				self.q_network.model.fit(state,q_value_target,batch_size=None,epochs=1, verbose=0)
				total_updates+=1
				if total_updates % 10000 == 0:
					print("Saving model at",total_updates)
					model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,total_updates) 
					filepath = os.path.join(save_dir, model_name)
					self.q_network.model.save(model_name)
				state = next_state

			# total_t_iter+=t_iter#
			# print("Total iterations is",total_t_iter)
			print("Total updates ",total_updates)

			print('----------------')

			# if ep_terminate==True:

				# break
		print("Saving final model at",total_updates)
		model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,total_updates) 
		filepath = os.path.join(save_dir, model_name)
		self.q_network.model.save(model_name)
		print("Total success is ",success_count)


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
			test_model = load_model(filepath)
			for e in range(num_episodes):
				total_epi_reward = 0
				print('Episode no ',e+1)
				test_state = self.env.reset()
				test_state = np.reshape(test_state,[1,self.state_size])	

				for t_iter in range(self.iterations):

					test_action = self.epsilon_greedy_policy(test_model.predict(test_state),0) 					

					self.env.render()
					test_next_state, test_reward, test_done, info = self.env.step(test_action)
					test_next_state = np.reshape(test_state,[1,self.state_size])	
					total_epi_reward+=test_reward
					
					# if (self.terminate==0 and state[0][0]==0.5) or (self.terminate==1 and t_iter==200):
					# 	print("success")
					# 	ep_terminate = True
					# 	break

					if test_done:
						print("Model %d" %(i))
						print("Episode finished after {} iterations with episodic reward %d and %d success".format(t_iter+1)%(total_epi_reward,success_count)) 
						plot_mat[e,(i/10000) - 1] = total_epi_reward
						break			
					test_state = test_next_state

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
	# dqn_agen.test()
	
	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

