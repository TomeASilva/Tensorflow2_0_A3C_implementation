from AC3_tf_2_0_Lunar_Landing import GlobalAgent, WorkerAgent
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import pickle

     

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

agent_configuration = {

    "action_space_bounds":[-1, 1],
    "action_space_size":2,
    "number_iter":100,
    "max_steps":2000,
    "n_episodes_per_cycle":1,
    
    "env_name":"LunarLanderContinuous-v2",
    "state_space_size":8,
}

def run_search (agent_configuration, hyperparameters):
    number_of_workers = 4
    params_queue = Manager().Queue(1)
    gradient_queue = Manager().Queue()
    current_iter = Manager().Value("i", 0)
    average_reward_queue = Queue(1)
    

    global_agent = GlobalAgent(**agent_configuration, 
                            **hyperparameters,
                            gradient_queue=gradient_queue,
                            parameters_queue = params_queue,
                            current_iter=current_iter,
                            name="Global Agent", 
                            record_statistics=False,
                            average_reward_queue=average_reward_queue)
    
    
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
    return average_reward_queue.get()

if __name__ == "__main__":

    with open("./parameter_search/current_hyperparameter.pickle", "rb") as file:
        current_hyperparameter = pickle.load(file)
    with open("./parameter_search/parameter_range.pickle", "rb") as file :
        hyperparameter_range = pickle.load(file)

    with open("./parameter_search/current_hyperparameter_index.pickle", "rb") as file:
        current_hyperparameter_index = pickle.load(file)
    with open("./parameter_search/max_reward.pickle", "rb") as file:
        max_reward_index = pickle.load(file)
    
    

    while current_hyperparameter != "gradient_clipping" and current_hyperparameter_index != len(hyperparameter_range["gradient_clipping"]) - 1:
        agent_configuration = {
                    "action_space_bounds":[-1, 1],
                    "action_space_size":2,
                    "number_iter":100,
                    "max_steps":2000,
                    "n_episodes_per_cycle":1,
                    "env_name":"LunarLanderContinuous-v2",
                    "state_space_size":8,
                }
        hyperparameters = {}
        
        for key, hyperparameter in hyperparameter_range.items():
            hyperparameters[key] = hyperparameter_range[key][max_reward_index[key]]
            
        reward = run_search(agent_configuration, hyperparameters)
        
        print(reward)
        exit()
        
        
        
        
        
    
    
    
                    
