###This file explains the porpuse of the each pickle and csv file within this folder.
___
- *current_hyperparameter_index.pickle* : this file stores the index of the value (int) for the current hyperparameter being searched.  
- *current_hyperparameter.pickle*: this file stores the name of the current hyperparameter(string) being searched
- *hyperparameter_list.pickle* : This file stores a list of strings, being that each string represents a name of one hyperparameter being searched
- *parameter_range.pickle* : this file stores a dictionary that defines a grid search, each key has as name  an hyperparameter that will be searched,and has as value a list of the values for that hyperparameter
- *max_reward.pickle*: This file stores a float with the max average reward find until the current moment

#####Examples: 

        current_hyperparameter = "entropy"
    
        current_hyperparameter_index = 0

        hyperparameter_list = [
                    "trunk_config",
                    "actor_mu_config",
                    "actor_cov_config", 
                    "critic_config",
                    "actor_optimizer",
                    "critic_optimizer",
                    "entropy", 
                    "gamma",
                    "gradient_clipping",
                    "end"]

        hyperparameter_range = {
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
        max_reward = 165.4


Knowing the structure of each python data structure we can now see its' use in the context of hyperparameter search. ```current_hyperparameter``` identifies what hyperparameter from the list of all the parameters we are searching in, ```current_hyperparameter_index```, tells us the position of the the value we are using at the current run within the list of possible values for that hyperparameter i.e. In our grid ```hyperparmeter_range```, we may be changing the ```"entropy"``` but we need to know what was the last value from ```entropy``` tested and what value should follow, that is the porpuse of ```current_hyperparameter_index```.

**Finally we have the csv file:**

This file stores the information of each grid search run. Each entry will have the following elements:
>- Average_reward
>- trunk_config
>- actor_mu_config
>- critic_config
>- actor_optimizer
>- actor_learning_rate
>- critic_learning_rate
>- entropy
>- gamma
>- gradient_clipping

Entries are reapeted in the pattern above, each repetition corresponds to a diferent run, therefor it will have different values for each hyperparameter.
The search happens in away so that every time we move from tuning one hyperparameter to the next, the previous hyperparameter gets locked in the value that led to highest averager reward for that hyperparameter. Following that reasoning the last search (```gradient_clipping```), will be done with the rest of hyperparameters set to the values that led to the higheste average rewards for each one of those hyperparameters. 