import pickle
with open("./parameter_search/hyperparameter_list.pickle", "rb") as file:
    mlist = pickle.load(file)
    
print(mlist)