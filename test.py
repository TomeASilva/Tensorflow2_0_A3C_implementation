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



agent_configuration = {
    "trunk_config":[trunk_config]

,
    "actor_mu_config": [mu_head_config],
    "actor_cov_config":[cov_head_config], 
    "critic_config": [critic_net_config],
    "actor_optimizer": [tf.keras.optimizers.SGD(learning_rate=0.001), tf.keras.optimizers.Adam(learning_rate=0.001)],
    "critic_optimizer": [tf.keras.optimizers.SGD(learning_rate=0.01), tf.keras.optimizers.Adam(learning_rate=0.001)],
    "entropy":[0.01, 0.02, 0.03],
    "gamma":[0.99, 0.98, 0.97],
    "gradient_clipping": [0.5, 0.8, 0.9, 1.0],
}




# with open("./parameter_search/max.json", "w") as jf:
#     json.dump(agent_configuration, jf)

# with open ("./parameter_search/parameter_range.pickle", "wb") as f:
#     pickle.dump(agent_configuration, f)


# hyperparameter_list = []
# for key, value in agent_configuration.items():
#     hyperparameter_list.append(key)


# with open ("./parameter_search/hyperparameter_list.pickle", "wb") as f:
#     pickle.dump(hyperparameter_list, f)


# hyperparameter = "trunk_config"
# with open ("./parameter_search/current_hyperparameter.pickle", "wb") as f:
#     pickle.dump(hyperparameter, f)

# index_current_hyperparameter = 0
# with open ("./parameter_search/current_hyperparameter_index.pickle", "wb") as f:
#     pickle.dump(index_current_hyperparameter, f)


def hyperpameter_search():
    with open("./parameter_search/current_hyperparameter.pickle", "rb") as file:
        current_hyperparameter = pickle.load(file)
    
    with open("./parameter_search/current_hyperparameter_index.pickle", "rb") as file:
        current_hyperparameter_index = pickle.load(file)
        
    with open("./parameter_search/parameter_range.pickle", "rb") as file :
        hyperparameter_range = pickle.load(file)
        
        
    print(hyperparameter_range)
hyperpameter_search()
    
    
    
    