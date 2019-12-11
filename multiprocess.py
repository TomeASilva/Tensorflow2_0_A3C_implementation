import random
import time
from multiprocessing import Manager, Process

class Agent():
    def __init__(self, params_queue, gradient_queue, iter_number):
        self.params = {"mu": random.randint(1, 10),
                       "cov": random.randint(1, 10),
                       "critic": random.randint(1, 10),
            }
        self.params_queue = params_queue
        self.gradient_queue = gradient_queue
        self.iter_number = iter_number
    


class Worker(Agent):
    def __init__(self, name, params_queue, gradient_queue, iter_number):
        Agent.__init__(self, params_queue, gradient_queue, iter_number)
        self.name = "Worker_" + str(name)
        
      
    def training_loop(self):
            for _ in range (15):
                
                time.sleep(random.randint(1,3))
                self.params = self.params_queue.get()
                print(f"{self.name} got params from global")
                self.params_queue.put(self.params)
                
                
                gradients = {
                    "mu": random.randint(1, 10),
                    "cov": random.randint(1, 10),
                    "critic": random.randint(1, 10)}
                
                self.gradient_queue.put((self.name,gradients))
                print(f"{self.name} added gradients to the Queue")
                self.iter_number.value += 1

        
        
        
    

class GlobalAgent(Agent):
    def __init__(self, params_queue, gradient_queue, iter_number, max_iter):
        Agent.__init__(self, params_queue, gradient_queue, iter_number)
        self.max_iter = max_iter
        self.params_queue.put(self.params)
        
    def learning_from_workers(self):
        
        while self.iter_number.value < self.max_iter:
            
            gradients = self.gradient_queue.get()
            print(f"Updating global with gradients from {gradients[0]}")
            for key, value in gradients[1].items():
                self.params[key] -= value
            
            self.params_queue.get()
            self.params_queue.put(self.params)
            print("Global params updated")
            
if __name__ == '__main__':

    params_queue = Manager().Queue(1)
    gradients_queue = Manager().Queue(1)
    iter_number = Manager().Value("i", 0)
    
    global_agent = GlobalAgent(params_queue,gradients_queue, iter_number, 60)
    
    workers = [Worker(_, params_queue, gradients_queue, iter_number) for _ in range(4)]
    
    p1 = Process(target=global_agent.learning_from_workers)
    p1.start()
    
    processes = []
    for worker in workers:
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    p1.join()
    
    for process in processes:
        process.join()
        
    
    print("Simulation Run")
    
    # manager = Manager()
    # save_que = manager.Queue()
    # file_ = "file"
    # save_p = Process(target=save_data, args=(save_que, file_))
    # save_p.start()
    # produce_data(save_que)
    # save_p.join()