import os
import datetime
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow_probability as tfp
import gym
import numpy as np
import time
import random
from collections import deque
from typing import Tuple, List
from multiprocessing import Manager, Process

print(tf.__version__)
class ep_buffer:
    """
    Class that stores the state transition information of an episode
    """

    def __init__(self):
        self.memory = deque()

    def add_transition(self, transition: Tuple) -> None:
        """
        Arguments:
        transition -> Tuple (s, a, s', reward)
        """

        self.memory.append(transition)

    @staticmethod
    def compute_Qsa(rewards, gamma: float) -> List:  # O(n^2)
        """
        Computes the sample value function (ground truth) for every single state action pair of an episode

        Arguments:
        rewards -> object that contain all the rewards from the episode from t = 0 to t = len(rewards)
        gamma -> float, discount factor for the rewards

        Returns:
        Qsa -> List

        """
        Qsa = []
        for i in range(len(rewards)):
            partial_Qsa = 0
            t = 0
            for j in range(i, len(rewards)):

                partial_Qsa += rewards[j] * (gamma ** t)
                t += 1

            Qsa.append(partial_Qsa)

        return Qsa

    def unroll_memory(self, gamma):
        """
        Unrolls the states transitions information so that states , actions, next_states, rewards and Qsa's
        are separeted into different numpy arrays

        Returns:
        states -> numpy array (state dimension, num of state transitions)
        actions -> numpy array (action dimension, num of state transitions)
        next_states -> numpy array (state dimension, num of state transitions)
        rewards -> numpy array (num of state transitions, )
        qsa -> numpy array (num of state transitions, )
        """

        states, actions, next_states, rewards = zip(*self.memory)

        qsa = self.compute_Qsa(rewards, gamma)
        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)
        rewards = np.asarray(rewards)
        qsa = np.asarray(qsa, dtype=np.float32).reshape(-1, 1)

        # print(f"States: {states.shape}")
        # print(f"actions: {actions.shape}")
        # print(f"next_states: {next_states.shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"qsa: {qsa.shape}")
        self.memory = deque()
        return states, actions, next_states, rewards, qsa

    

    
def build_networks(layer_sizes, activations, input):
    num_layers = len(layer_sizes)
    output = keras.layers.Dense(units=layer_sizes[0], activation=activations[0], kernel_initializer='glorot_uniform')(input)
    for i in range(1, num_layers):
        output = keras.layers.Dense(units=layer_sizes[i], activation=activations[i], kernel_initializer='glorot_uniform')(output)
    
    return output
    
    
def build_model(input, output, name):
    return keras.Model(input, output, name=name)


class Env:
    state_space_size = 0 
    action_space_size = 0
    def __init__(self, env_name, action_space_size, state_space_size):
        Env.state_space_size = state_space_size
        Env.action_space_size = action_space_size
        self.name = env_name
    
    

class Agent(ep_buffer, Env):
    def __init__(self,
                 actor_mu,
                 actor_cov,
                 critic, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env,
                 state_space_size,
                 gradient_queue,
                 parameters_queue,
                 current_iter): 
        
        ep_buffer.__init__(self)
        Env.__init__(self, env, action_space_size, state_space_size)

     
        self.actor_mu = actor_mu
        self.actor_cov = actor_cov
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.entropy = entropy
        self.action_bounds = action_space_bounds
        self.number_iter = number_iter
        self.max_steps = max_steps
        self.n_episodes_per_cycle = n_episodes_per_cycle        
        self.gamma = gamma
        self.env = gym.make(env)
        self.gradient_queue = gradient_queue
        self.parameters_queue = parameters_queue
        
        index_last_layer = len(actor_cov.layers) -1
        self.current_parameters = {"mu": actor_mu.trainable_variables[0].numpy(),
                           "cov": actor_cov.get_layer(index=index_last_layer).trainable_variables[0].numpy(),
                           "critic": critic.trainable_variables[0].numpy()
            
        }
        self.variables =  {"mu": actor_mu.trainable_variables,
                           "cov": actor_cov.get_layer(index=index_last_layer).trainable_variables,
                           "critic": critic.trainable_variables
            
        }
        self.current_iter = current_iter
    
    def collect_episodes(self, number_ep, max_steps, render=False):
        total_steps = 0
        total_reward = 0
        for ep in range(number_ep):
            prev_observation = self.env.reset()
            steps = 0
            done = False
            while done == False and steps <= max_steps:
                if render:
                    self.env.render()

                action = self.take_action(prev_observation.reshape(1, -1))
                action = action.numpy().reshape(self.action_space_size,)
                observation, reward, done, _ = self.env.step(action)
                steps += 1

                if steps == max_steps and not done:
                    
                    reward += self.state_value(observation.reshape(1, -1))
                

                self.add_transition((prev_observation, action, observation, reward))
                prev_observation = observation
                total_reward += reward

            total_steps += steps

            self.env.close()
        return total_reward
    @tf.function
    def state_value(self, state):
        value = float(self.critic(state).numpy()[0])
        return value
        
        
        
    @tf.function
    def take_action(self, state):
        # state numpy
        
        mu = self.actor_mu(state)
        cov = self.actor_cov(state)
        
        probability_density_func = tfp.distributions.Normal(mu, cov)
        action = tf.clip_by_value(probability_density_func.sample(1), self.action_bounds[0], self.action_bounds[1])
       
        # print(f"Action {action}\n Type: {action.shape}")
        return action
    
        
class GlobalAgent(Agent):
    def __init__(self,
                 actor_mu,
                 actor_cov,
                 critic, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env,
                 state_space_size,
                 gradient_queue,
                 parameters_queue,
                 current_iter):
        Agent.__init__(self,
                 actor_mu,
                 actor_cov,
                 critic, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env,
                 state_space_size,
                 gradient_queue,
                 parameters_queue,
                 current_iter)
        
      
        self.parameters_queue.put(self.current_parameters)
        
    def training_loop(self):
        
        while self.current_iter.value < self.number_iter:
            gradients = self.gradient_queue.get()
            for key, value in gradients.items():
                
                self.actor_optimizer.apply_gradients(zip(value, self.variables[key]))
                self.parameters_queue.get()
                
                self.current_parameters = {"mu": actor_mu.trainable_variables[0].numpy(),
                                   "cov": actor_cov.get_layer(index=index_last_layer).trainable_variables[0].numpy(),
                                   "critic": critic.trainable_variables[0].numpy()
                    
        }
                self.parameters_queue.put(self.parameters)
        
            if self.current_iter.value % 100:
                reward = self.collect_episodes(1, 2000)
                print(f"Reward at iter: {self.current_iter.value} is {reward}")
 
        
class WorkerAgent(Agent):
   
    def __init__(self,
                 actor_mu,
                 actor_cov,
                 critic, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env,
                 state_space_size,
                 gradient_queue,
                 parameters_queue,
                 current_iter): 
        
        Agent.__init__(self,
                        actor_mu,
                        actor_cov,
                        critic, 
                        actor_optimizer,
                        critic_optimizer,
                        entropy,
                        action_space_bounds,
                        action_space_size,
                        number_iter,
                        max_steps,
                        n_episodes_per_cycle,
                        gamma,
                        env,
                        state_space_size,
                        gradient_queue,
                        parameters_queue,
                        current_iter)

        
    def training_loop(self):
    
        rewards_collection = deque(maxlen=100)
        while self.current_iter.value < self.number_iter:
            
            self.update_variables()
            
            reward_ep = self.collect_episodes(self.n_episodes_per_cycle, self.max_steps)
            states, actions, next_states, rewards, qsa = self.unroll_memory(self.gamma)
            
            # states = tf.convert_to_tensor(states)
            # actions = tf.convert_to_tensor(actions)
            # rewards = tf.convert_to_tensor(rewards)
            gradients = train_step(states, actions, qsa)
            
            self.gradient_queue.put(gradients)
          
            rewards_collection.append(reward_ep)
            
            self.current_iter.value += 1

            # if iter % 100 == 0:
            
            #     average_reward = sum(rewards_collection)/100
            #     print(f"Average reward at ep {iter} is -> {average_reward}")
    

            # if iter % 1000 == 0:
            #     self.collect_episodes(1, 2000, render=True)
    @tf.function
    def update_variables(self):
        
        self.new_params = self.parameters_queue.get()
        for key, value in self.new_params.items():
            
            self.variables[key].assign(value)
           
                
        self.parameters_queue.put(self.new_params)
                
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 8]), tf.TensorSpec(shape=[None, 2]),tf.TensorSpec(shape=[None, 1])))
    def train_step(self, states, actions, Qsa):
        with tf.GradientTape(persistent=True) as tape:
            # Actor
            mu = self.actor_mu(states)
            cov = self.actor_cov(states)
            advantage_function = Qsa - self.critic(states)
            probability_density_func = tfp.distributions.Normal(mu, cov)
            entropy = probability_density_func.entropy()
            log_probs = probability_density_func.log_prob(actions)
            expected_value = tf.multiply(log_probs, advantage_function)
            expected_value_with_entropy = expected_value + entropy * self.entropy
            actor_cost = -tf.reduce_mean(expected_value_with_entropy)
            
            # Critic
            critic_cost = tf.losses.mean_squared_error(Qsa, self.critic(states))
        
        gradients_mu = tape.gradient(actor_cost, self.actor_mu.trainable_variables)
        last_layer_index= len(self.actor_cov.layers) - 1
        
        gradients_cov = tape.gradient(actor_cost, self.actor_cov.get_layer(index=last_layer_index).trainable_variables)
        gradients_critic = tape.gradient(critic_cost, self.critic.trainable_variables)
        gradients = {"mu": gradients_mu,
                     "cov": gradients_cov,
                     "critic": gradients_critic,
        }
        return gradients
                
            
        


trunk_parameters = {
   
    "layer_sizes": [100, 100],
    "activations": [ "relu", "relu"],
}



mu_head_parameters = {
    "layer_sizes":[2],
    "activations": [tf.nn.tanh],
    "input": trunk}

cov_head_parameters = {
    "layer_sizes":[2],
    "activations": [tf.nn.softplus],
    "input": trunk
    }

critic_net_parameters = {
    "layer_sizes":[100, 10, 1],
    "activations": ["relu", "relu", "linear"],
    "input": state_input
    }

mu_head = build_networks(**mu_head_parameters)
cov_head = build_networks(**cov_head_parameters)
critic_net = build_networks(** critic_net_parameters)

mu_model = build_model(state_input,mu_head, "actor_mu")
cov_model = build_model(state_input, cov_head, "actor_cov")
critic_model = build_model(state_input, critic_net, "critic")

number_of_workers = 4

agent_configuration = {
    "actor_mu":mu_model,
    "actor_cov":cov_model, 
    "critic":critic_model,
    "actor_optimizer":tf.keras.optimizers.Adam(0.0001),
    "critic_optimizer":tf.keras.optimizers.Adam(0.01),
    "entropy":0.01,
    "action_space_bounds":[-1,1],
    "action_space_size":2,
    "number_iter":200,
    "max_steps":2000,
    "n_episodes_per_cycle":1,
    "gamma":0.99,
    "env":"LunarLanderContinuous-v2",
    "state_space_size":8
}

if __name__ == '__main__':
    
    number_of_workers = 4
    params_queue = Manager().Queue(1)
    gradient_queue = Manager().Queue(1)
    current_iter = Manager().Value("i", 0)
    
    global_agent = GlobalAgent(**agent_configuration,
                               gradient_queue=gradient_queue,
                               parameters_queue = params_queue,
                               current_iter=current_iter)
    
    workers = [WorkerAgent(**agent_configuration, gradient_queue=gradient_queue, parameters_queue = params_queue, current_iter=current_iter) for _ in range(number_of_workers)]

    p1 = Process(target=global_agent.training_loop)
    p1.start()
    for worker in workers:
        
        p = Process(target=worker.traning_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("Simulation Run")
        
        
        
# time_start = time.time()

# agent.training_loop()

# time_elapsed = time.time() - time_start
# print(f"Elapsed time {time_elapsed}")