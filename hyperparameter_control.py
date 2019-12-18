import json
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




def refresh_search():
    #refresh range
    agent_configuration ={
        "trunk_config":[trunk_config],
        "actor_mu_config": [mu_head_config],
        "actor_cov_config":[cov_head_config], 
        "critic_config": [critic_net_config],
        "actor_optimizer": [tf.keras.optimizers.SGD(learning_rate=0.001), tf.keras.optimizers.Adam(learning_rate=0.001)],
        "critic_optimizer": [tf.keras.optimizers.SGD(learning_rate=0.01), tf.keras.optimizers.Adam(learning_rate=0.001)],
        "entropy":[0.01, 0.02, 0.03],
        "gamma":[0.99, 0.98, 0.97],
        "gradient_clipping": [0.5, 0.8, 0.9, 1.0],
    }
    with open ("./parameter_search/parameter_range.pickle", "wb") as f:
        pickle.dump(agent_configuration, f)
        
    #refresh current_hyperparameter
    
    hyperparameter = "trunk_config"
    with open ("./parameter_search/current_hyperparameter.pickle", "wb") as f:
        pickle.dump(hyperparameter, f)
        
    #refresh current_hyperparameter_index 
    index_current_hyperparameter = 0
    with open("./parameter_search/current_hyperparameter.pickle", "wb") as f:
        pickle.dump(index_current_hyperparameter, f)
    
    # refresh hyper_max_reward
    
    max_reward_index = {
                    "trunk_config": 0,
                    "actor_mu_config": 0,
                    "actor_cov_config": 0, 
                    "critic_config": 0,
                    "actor_optimizer": 0,
                    "critic_optimizer": 0,
                    "entropy": 0, 
                    "gamma": 0,
                    "gradient_clipping": 0,}
    
    with open ("./parameter_search/max_reward.pickle", "wb") as f:
        pickle.dump(max_reward_index, f)
    

if __name__ == "__main__":
    refresh_search()

    
    