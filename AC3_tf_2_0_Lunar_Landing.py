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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
# print(tf.__version__)

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
    output = keras.layers.Dense(units=layer_sizes[0], activation=activations[0], kernel_initializer='glorot_normal')(input)
    for i in range(1, num_layers):
        output = keras.layers.Dense(units=layer_sizes[i], activation=activations[i], kernel_initializer='glorot_normal')(output)
    
    return output
    
    
def build_model(input, output, name):
    return keras.Model(input, output, name=name)

    

class Agent(ep_buffer):
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping,
                 gradient_queue,
                 parameters_queue,
                 current_iter, 
                 name,
                 record_statistics):
        ep_buffer.__init__(self)
        

        self.gradient_clipping = gradient_clipping
        self.trunk_config = trunk_config
        self.actor_mu_config = actor_mu_config
        self.actor_cov_config = actor_cov_config
        self.critic_config = critic_config
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.entropy = entropy
        self.action_space_bounds = action_space_bounds
        self.action_space_size = action_space_size
        self.action_bounds = action_space_bounds
        self.number_iter = number_iter
        self.max_steps = max_steps
        self.n_episodes_per_cycle = n_episodes_per_cycle        
        self.gamma = gamma
        self.env_name = env_name
        self.gradient_queue = gradient_queue
        self.parameters_queue = parameters_queue
        self.name = name
        self.current_iter = current_iter
        self.record_statistics = record_statistics
        
    def build_models(self):
        
        self.input = keras.Input(shape=(8), name="state")
        self.trunk = build_networks(**self.trunk_config, input=self.input)
        mu_head = build_networks(**self.actor_mu_config, input=self.trunk)
        cov_head = build_networks(**self.actor_cov_config, input=self.trunk)
        critic = build_networks(**self.critic_config, input=self.input)
        self.actor_mu = build_model(self.input, mu_head, "actor_mu")
        self.actor_cov = build_model(self.input, cov_head, "actor_cov")
        self.critic = build_model(self.input, critic, "critic")
        
        index_last_layer = len(self.actor_cov.layers) -1
        
 
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
        
                           }
    
        self.variables =  {"mu": self.actor_mu.trainable_variables,
                           "cov": self.actor_cov.get_layer(index=index_last_layer).trainable_variables,
                           "critic": self.critic.trainable_variables
            
        }
        
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
        # tf.print("mu-------")
        # tf.print(mu)
        # tf.print("Cov------")
        # tf.print(cov)        
        probability_density_func = tfp.distributions.Normal(mu, cov)
        action = probability_density_func.sample(1)
        # tf.print("Action")
        # tf.print(action)
        action = tf.clip_by_value(action, self.action_bounds[0], self.action_bounds[1])
       
        # print(f"Action {action}\n Type: {action.shape}")
        return action
    
        
class GlobalAgent(Agent):
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping,
                 gradient_queue,
                 parameters_queue,
                 current_iter,
                 name,
                 record_statistics):
        
        Agent.__init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping,
                 gradient_queue,
                 parameters_queue,
                 current_iter, 
                 name,
                 record_statistics)
        
      
    def training_loop(self):
        self.build_models()
        self.env = gym.make(self.env_name)
        if self.record_statistics: 
            self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.current_pass = 0 
        self.parameters_queue.put(self.current_parameters, block=True, timeout=30)
        time.sleep(2)
        while self.current_iter.value < self.number_iter:
            self.iter = self.current_iter.value
            self.parameters_queue.get(block=True, timeout=30)
       
            gradients = self.gradient_queue.get(block=True, timeout=30)
            # print(f"Global Agent got gradients from {gradients['name']} --Iter: {self.iter}")
           
            #check the gradients
            if self.record_statistics :
                with self.writer.as_default():
                    for key, grandient_list in gradients.items():
                        if key != "name": 
                            for gradient, variable in zip(grandient_list, self.variables[key]):
                                tf.summary.histogram(f"{self.name}_{str(key)}_{variable.name}", gradient, self.current_pass)
                        
                
            for key, value in gradients.items():
                if key != "name":
                    self.actor_optimizer.apply_gradients(zip(value, self.variables[key]))
            index_last_layer = len(self.actor_cov.layers) - 1
            
            self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
        
                           }
            self.parameters_queue.put(self.current_parameters, block=True, timeout=30)
            # print(f"New weights available from {gradients['name']} gradients --Iter: {self.iter}")

            self.current_pass += 1
            if self.iter % 100 == 0:
                reward = self.collect_episodes(1, 2000, render=True)
                print(f"Reward at iter: {self.iter} is {reward}")
 
        
class WorkerAgent(Agent):
   
    def __init__(self,
                 trunk_config,
                 actor_mu_config,
                 actor_cov_config,
                 critic_config, 
                 actor_optimizer,
                 critic_optimizer,
                 entropy,
                 action_space_bounds,
                 action_space_size,
                 number_iter,
                 max_steps,
                 n_episodes_per_cycle,
                 gamma,
                 env_name,
                 state_space_size,
                 gradient_clipping,
                 gradient_queue,
                 parameters_queue,
                 current_iter,
                 name,
                 record_statistics): 
        
        Agent.__init__( self,
                        trunk_config,
                        actor_mu_config,
                        actor_cov_config,
                        critic_config, 
                        actor_optimizer,
                        critic_optimizer,
                        entropy,
                        action_space_bounds,
                        action_space_size,
                        number_iter,
                        max_steps,
                        n_episodes_per_cycle,
                        gamma,
                        env_name,
                        state_space_size,
                        gradient_clipping,
                        gradient_queue,
                        parameters_queue,
                        current_iter, 
                        name,
                        record_statistics)

        
    def training_loop(self):
        if self.record_statistics: 
            self.writer = tf.summary.create_file_writer(f"./summaries/worker/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.build_models()
        self.env = gym.make(self.env_name)
        self.number_passes = 0 # to count the number of iteration done within worker
        
        rewards_collection = deque(maxlen=100)
        while self.current_iter.value < self.number_iter:
            
            self.iter = self.current_iter.value
            # print(f"{self.parameters_queue.qsize()} variables waiting at {self.name} --Iter : {self.iter} ")
            self.update_variables()
            #check Weights after update
            if self.record_statistics:
                with self.writer.as_default():
                    for key, list_variables in self.variables.items():
                        for variable in list_variables:
                            tf.summary.histogram(f"{key}_{variable.name}", variable, step=self.number_passes)
                            
            # print(f"{self.name} syncronized its variables-- Iter: {self.iter}")
            reward_ep = self.collect_episodes(self.n_episodes_per_cycle, self.max_steps)
            states, actions, next_states, rewards, qsa = self.unroll_memory(self.gamma)
            
            # states = tf.convert_to_tensor(states)
            # actions = tf.convert_to_tensor(actions)
            # rewards = tf.convert_to_tensor(rewards)
            gradients = self.train_step(states, actions, qsa)
         
            #Cliping gradients
            for key, gradient in gradients.items():
                if key != "name":
                    gradients[key] = [tf.clip_by_norm(value, 0.1) for value in gradient]
                    
        
            self.gradient_queue.put(gradients)
            
      
          
            # print(f"Gradients from {self.name} available-- Iter: {self.iter}")
            rewards_collection.append(reward_ep)
            
            self.current_iter.value += 1
            self.number_passes += 1

            # if iter % 100 == 0:
            
            #     average_reward = sum(rewards_collection)/100
            #     print(f"Average reward at ep {iter} is -> {average_reward}")
    

            # if iter % 1000 == 0:
            #     self.collect_episodes(1, 2000, render=True)

    def update_variables(self):
        
        self.new_params = self.parameters_queue.get(block=True, timeout=30)
        self.parameters_queue.put(self.new_params)
        for key, value in self.new_params.items():
            for n, variable in enumerate(self.variables[key]):
                variable.assign(value[n])

        index_last_layer = len(self.actor_cov.layers) - 1 
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                        "cov": [variable.numpy() for variable in self.actor_cov.get_layer(index=index_last_layer).trainable_variables],
                        "critic": [variable.numpy() for variable in self.critic.trainable_variables]
    
                        }
    
                
    
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
                     "name": self.name,
        }
        
        return gradients
                
            

trunk_config = {
   
    "layer_sizes": [100, 100],
    "activations": [ "relu", "relu"],
}



mu_head_config = {
    "layer_sizes":[50, 40, 2],
    "activations": ["relu", "relu", "relu", "tanh"]
    }

cov_head_config = {
    "layer_sizes":[50, 40, 2],
    "activations": ["relu", "relu", "softplus"],
  
    }

critic_net_config= {
    "layer_sizes":[100, 64, 1],
    "activations": ["relu", "relu", "linear"],
    }


hyperparameters = { "trunk_config": trunk_config,
                    "actor_mu_config": mu_head_config,
                    "actor_cov_config":cov_head_config, 
                    "critic_config": critic_net_config,
                    "actor_optimizer": tf.keras.optimizers.SGD(learning_rate=0.001),
                    "critic_optimizer": tf.keras.optimizers.SGD(learning_rate=0.01),
                    "entropy":0.01,
                    "gamma":0.99,
                    "gradient_clipping": 0.5
                    }
# agent_configuration = {
#     "trunk_config": trunk_config,
#     "actor_mu_config": mu_head_config,
#     "actor_cov_config":cov_head_config, 
#     "critic_config": critic_net_config,
#     "actor_optimizer": tf.keras.optimizers.SGD(learning_rate=0.001),
#     "critic_optimizer": tf.keras.optimizers.SGD(learning_rate=0.01),
#     "entropy":0.01,
#     "action_space_bounds":[-1, 1],
#     "action_space_size":2,
#     "number_iter":10000,
#     "max_steps":2000,
#     "n_episodes_per_cycle":1,
#     "gamma":0.99,
#     "env_name":"LunarLanderContinuous-v2",
#     "state_space_size":8,
#     "gradient_clipping": 0.5,
# }

agent_configuration = {

    "action_space_bounds":[-1, 1],
    "action_space_size":2,
    "number_iter":10000,
    "max_steps":2000,
    "n_episodes_per_cycle":1,
    
    "env_name":"LunarLanderContinuous-v2",
    "state_space_size":8,
}


# "actor_optimizer":tf.keras.optimizers.RMSprop(0.01),
# "critic_optimizer":tf.keras.optimizers.RMSprop(0.1),
if __name__ == '__main__':
    
    number_of_workers = 4
    params_queue = Manager().Queue(1)
    gradient_queue = Manager().Queue(1)
    current_iter = Manager().Value("i", 0)
    
    # global_agent = GlobalAgent(**agent_configuration,
    #                            gradient_queue=gradient_queue,
    #                            parameters_queue = params_queue,
    #                            current_iter=current_iter,
    #                            name="Global Agent", 
    #                            record_statistics=False)
    
    global_agent = GlobalAgent(**agent_configuration, 
                               **hyperparameters,
                            gradient_queue=gradient_queue,
                            parameters_queue = params_queue,
                            current_iter=current_iter,
                            name="Global Agent", 
                            record_statistics=False)
    
    # workers = [WorkerAgent(**agent_configuration, **hyperparameters, gradient_queue=gradient_queue, parameters_queue = params_queue, current_iter=current_iter, name=f"Worker_{_}", record_statistics=False) for _ in range(number_of_workers)]
    workers = [WorkerAgent(**agent_configuration, **hyperparameters, gradient_queue=gradient_queue, parameters_queue = params_queue, current_iter=current_iter, name=f"Worker_{_}", record_statistics=False) for _ in range(number_of_workers)]
    
    processes = []
    p1 = Process(target=global_agent.training_loop)
    processes.append(p1)
    p1.start()

    for worker in workers:
        
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("Simulation Run")
        
        
        
# time_start = time.time()

# agent.training_loop()

# time_elapsed = time.time() - time_start
# print(f"Elapsed time {time_elapsed}")