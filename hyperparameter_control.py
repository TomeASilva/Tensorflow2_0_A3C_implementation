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
    """Refreshes the grid search parameters i.e defines the grid for the parameters search, sets current hyperparameter to trunk_config,
    and the current_hyperparameter_index to 0, each one of these variables is then dumped into a pickle file
    This function should be used, when we want to reset a hyperparameter search, either because we finished the a grid search, or because we wante to change the grid
    after running it for some time without the expected results"""
    
    hyperparameter_list = ["trunk_config",
                    "actor_mu_config",
                    "actor_cov_config", 
                    "critic_config",
                    "actor_optimizer",
                    "critic_optimizer",
                    "entropy", 
                    "gamma",
                    "gradient_clipping",
                    "end"]
    
    with open ("./parameter_search/hyperparameter_list.pickle", "wb") as file:
        pickle.dump(hyperparameter_list, file)
        
    agent_configuration ={
        "trunk_config":[trunk_config],
        "actor_mu_config": [mu_head_config],
        "actor_cov_config":[cov_head_config], 
        "critic_config": [critic_net_config],
        "actor_optimizer":[tf.keras.optimizers.SGD(learning_rate=0.01), tf.keras.optimizers.SGD(learning_rate=0.001), 
                           tf.keras.optimizers.SGD(learning_rate=0.0001), tf.keras.optimizers.Adam(learning_rate=0.01), 
                           tf.keras.optimizers.Adam(learning_rate=0.001), tf.keras.optimizers.Adam(learning_rate=0.0001), 
                           tf.keras.optimizers.RMSprop(learning_rate=0.01), 
                           tf.keras.optimizers.RMSprop(learning_rate=0.001),tf.keras.optimizers.RMSprop(learning_rate=0.0001)],
        "critic_optimizer":[tf.keras.optimizers.SGD(learning_rate=0.01), tf.keras.optimizers.SGD(learning_rate=0.001),
                            tf.keras.optimizers.SGD(learning_rate=0.0001), tf.keras.optimizers.Adam(learning_rate=0.01),
                            tf.keras.optimizers.Adam(learning_rate=0.001), tf.keras.optimizers.Adam(learning_rate=0.0001),
                            tf.keras.optimizers.RMSprop(learning_rate=0.01), tf.keras.optimizers.RMSprop(learning_rate=0.001),
                            tf.keras.optimizers.RMSprop(learning_rate=0.0001)],
        "entropy": [0.005, 0.01, 0.02, 0.07, 0.1, 0.12],
        "gamma":[0.99, 0.90, 0.80, 0.75, 0.70, 0.65, 0.60],
        "gradient_clipping": [0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0],
    }
    with open ("./parameter_search/parameter_range.pickle", "wb") as f:
        pickle.dump(agent_configuration, f)
        
    #refresh current_hyperparameter
    
    hyperparameter = "trunk_config"
    with open ("./parameter_search/current_hyperparameter.pickle", "wb") as f:
        pickle.dump(hyperparameter, f)
        
    #refresh current_hyperparameter_index 
    index_current_hyperparameter = 0
    with open("./parameter_search/current_hyperparameter_index.pickle", "wb") as f:
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

    
    