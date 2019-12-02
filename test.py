import tensorflow as tf
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import Tuple, List
from collections import deque
import numpy as np
import gym
import time



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


def run_random_episodes(number_ep, max_steps):
    env = gym.make('LunarLanderContinuous-v2')
    total_steps = 0
    for ep in range(number_ep):
        prev_observation = env.reset()
        steps = 0
        done = False
        while done == False and steps < max_steps:
            env.render()
            action = env.action_space.sample()
            print(action)
            print(type(action))
            print(action.shape)

            observation, reward, done, _ = env.step(action)
            steps += 1
            # buffer.add_transition((prev_observation, action, observation, reward))
            print(reward)
            prev_observation = observation
        total_steps += steps
        env.close()

    return total_steps


def build_network(network_name: str, layer_sizes: List, layer_activations: List):
    """
    Arguments:
    network_name -> string
    layer_sizes -> list, the number o units per each layer len(layer_sizes) = n layers, including input layer
    layer_activations -> list, the activation for every layer, including input layer ex:
    3 layers: 1 input: 1 hidden : 1 output -> [None, "relu", "linear"]

    Returns:
    output -> numpy.array -- the forwardprop result : shape [number_training examples, number of output units]
    params -> params -- list with tf.Variable objects created in the network: weights and biases
    inputs -> tf tensor -- tensor placeholder for the input layer of the network
    """
    # Create layers
    # forward propagation
    num_layers = len(layer_sizes)

    with tf.variable_scope(network_name):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, layer_sizes[0]], name="state")

        output = tf.layers.Dense(units=layer_sizes[1], activation=layer_activations[1], kernel_initializer=tf.initializers.glorot_normal(),
                                 name="Layer_1")(inputs)

        for i in range(2, num_layers):

            output = tf.layers.Dense(units=layer_sizes[i], activation=layer_activations[i], kernel_initializer=tf.initializers.glorot_normal(),
                                     name=f"Layer_{i}")(output)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)

        return output, params, inputs


def build_AC3_graph(actor_mu_parameters, actor_covariance_parameters, critic_parameters, action_space_size, entropy_param, action_space_bounds, actor_optimizer, critic_optimizer):
    # Actor network
    mu, mu_variables, state_placeholder_mu = build_network(**actor_mu_parameters)

    covariance, covariance_variables, state_placeholder_covariance = build_network(**actor_covariance_parameters)

    # Critic Network

    v, v_variables, state_placeholder_v = build_network(**critic_parameters)

    Qsa_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Qsa")

    with tf.variable_scope("Actor_Ops"):
        with tf.variable_scope("Advantage_Function"):
            advantage_function = Qsa_placeholder - v

        actions_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, action_space_size], name="Actions")
        # Actor cost function
        with tf.variable_scope("Cost"):
            probability_density_func = tf.distributions.Normal(mu, covariance)
            log_probs = probability_density_func.log_prob(actions_placeholder)
            entropy = probability_density_func.entropy()
            expected_value = tf.multiply(log_probs, advantage_function)

            expected_value_w_entropy = expected_value + entropy * entropy_param

            actor_cost = - tf.reduce_sum(expected_value_w_entropy)
        with tf.variable_scope("Train"):
            op_actor = actor_optimizer.minimize(actor_cost)

        with tf.variable_scope("Take_Action"):

            action = tf.clip_by_value(probability_density_func.sample(1), action_space_bounds[0], action_space_bounds[1])

    with tf.variable_scope("Critic_Ops"):
        with tf.variable_scope("Cost"):

            critic_cost = tf.losses.mean_squared_error(Qsa_placeholder, v)
        with tf.variable_scope("Train"):
            op_critic = critic_optimizer.minimize(critic_cost)

    actor = {
        "cost": actor_cost,
        "state_placeholder_mu": state_placeholder_mu,
        "state_placeholder_covariance": state_placeholder_covariance,
        "actions_placeholder": actions_placeholder,
        "action": action,
        "op": op_actor
    }
    critic = {
        "cost": critic_cost,
        "state_placeholder_v": state_placeholder_v,
        "Qsa": Qsa_placeholder,
        "op": op_critic,
        "v": v
    }
    return actor, critic

# session = tf.Session()
# writer = tf.summary.FileWriter("./graph/")

# session.run(tf.global_variables_initializer())

# build_AC3_graph(actor_mu_parameters, actor_covariance_parameters, critic_parameters, 2, 0.01)

# writer.add_graph(session.graph)
# session.close()


class Agent(ep_buffer):
    def __init__(self,
                 actor_mu_parameters,
                 actor_covariance_parameters,
                 critic_parameters,
                 action_space_size,
                 action_space_bounds,
                 entropy_param,
                 environment,
                 max_steps,
                 n_episodes_per_cycle,
                 number_iterations,
                 gamma,
                 actor_optimizer,
                 critic_optimizer):
        super().__init__()

        actor, critic = build_AC3_graph(actor_mu_parameters,
                                        actor_covariance_parameters,
                                        critic_parameters,
                                        action_space_size,
                                        entropy_param,
                                        action_space_bounds,
                                        actor_optimizer,
                                        critic_optimizer)
        self.actor = actor
        self.critic = critic
        self.env = gym.make(environment)
        self.session = tf.Session()
        self.max_steps = max_steps
        self.n_episodes_per_cycle = n_episodes_per_cycle
        self.number_iterations = number_iterations
        self.gamma = gamma
        self.actor_optimizer = actor
        self.session.run(tf.global_variables_initializer())
        self.action_space_size = action_space_size

    def optimization_loop(self):
        rewards_collection = deque(maxlen=100)
        for iter in range(self.number_iterations):
            reward_ep = self.collect_episodes(self.n_episodes_per_cycle, self.max_steps)
            states, actions, next_states, rewards, qsa = self.unroll_memory(self.gamma)

            _, cost_critic = self.session.run([self.critic["op"], self.critic["cost"]], feed_dict={
                self.critic["state_placeholder_v"]: states,
                self.critic["Qsa"]: qsa
            })
            # run gradient descent
            _, cost_actor = self.session.run([self.actor["op"], self.actor["cost"]], feed_dict={
                self.actor["state_placeholder_mu"]: states,
                self.actor["state_placeholder_covariance"]: states,
                self.actor["actions_placeholder"]: actions,
                self.critic["state_placeholder_v"]: states,
                self.critic["Qsa"]: qsa
            })

            rewards_collection.append(reward_ep)
           
            if iter % 100 == 0:
                print(f"Iter {iter} Cost: {cost_actor}")
                average_reward = sum(rewards_collection)/100
                print(f"Average reward at ep {iter} is -> {average_reward}")
                tf.train.Checkpoint()

            if iter % 1000 == 0:
                self.collect_episodes(1, 2000, render=True)

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
                observation, reward, done, _ = self.env.step(action)
                steps += 1

                if steps == max_steps and not done:
                    print("crossed")
                    reward += float(self.session.run([self.critic["v"]], feed_dict={
                        self.critic["state_placeholder_v"]: observation.reshape(1, -1)
                    })[0])

                self.add_transition((prev_observation, action, observation, reward))
                prev_observation = observation
                total_reward += reward

            total_steps += steps

            self.env.close()
        return total_reward

    def take_action(self, state):
        # state numpy?
        action = self.session.run(self.actor["action"], feed_dict={
            self.actor["state_placeholder_mu"]: state,
            self.actor["state_placeholder_covariance"]: state
        })
        action = action.reshape(self.action_space_size,)
        # print(f"Action {action}\n Type: {action.shape}")
        return action


actor_mu_parameters = {
    "network_name": "Actor_mu",
    "layer_sizes": [8, 100, 100, 2],
    "layer_activations": [None, "relu", "relu", tf.nn.tanh]


}
actor_covariance_parameters = {
    "network_name": "Actor_covariance",
    "layer_sizes": [8, 100, 100, 2],
    "layer_activations": [None, "relu", "relu", tf.nn.softplus]

}
critic_parameters = {
    "network_name": "Critic",

    "layer_sizes": [8, 100, 100, 1],
    "layer_activations": [None, "relu", "relu", "linear"]
}

agent = Agent(actor_mu_parameters,
              actor_covariance_parameters,
              critic_parameters,
              action_space_size=2,
              action_space_bounds=[-1, 1],
              entropy_param=0.01,
              environment="LunarLanderContinuous-v2",
              max_steps=2000,
              n_episodes_per_cycle=1,
              number_iterations=200,
              gamma=0.99,
              actor_optimizer=tf.train.AdamOptimizer(0.0001),
              critic_optimizer=tf.train.AdamOptimizer(0.01))

time_start = time.time()
agent.optimization_loop()
time_elapsed = time.time() - time_start
print(f"Elapsed time {time_elapsed}")
# run_random_episodes(1, 3, )
