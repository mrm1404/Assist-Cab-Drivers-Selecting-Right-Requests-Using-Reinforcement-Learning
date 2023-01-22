# Import routines
import numpy as np
import random

# Defining constants
m = 5 # number of cities, ranges from 1 ... m
t = 24 # number of hours, ranges from 0 ... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

Time_matrix = np.load("TM.npy")

class CabDriver():

    def __init__(self):
        """define action space and state space and initialize state size"""
        self.action_space = [(i,j) for i in range(1,m+1) for j in range(1,m+1) if i!=j]+[(0,0)]
        self.state_space = [(i,j,k) for i in range(1,m+1) for j in range(0,t) for k in range(0,d)]
        self.state_size = m + t + d

        self.reset()


    ## Encoding state (or state-action) for NN input
    def encode_state(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. The vector is of size m + t + d i.e. 36"""
        loc_array = np.zeros(m)
        hour_array = np.zeros(t)
        day_array = np.zeros(d)

        loc_array[state[0]-1] = 1
        hour_array[state[1]] = 1
        day_array[state[2]] = 1

        encoded_state = np.concatenate((loc_array, hour_array, day_array), axis=None)

        return encoded_state


    def get_poisson(self, location):
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12) 
        if location == 3:
            requests = np.random.poisson(4) 
        if location == 4:
            requests = np.random.poisson(7) 
        if location == 5:
            requests = np.random.poisson(8)

        return requests


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. Use the table specified in the MDP and complete for rest of the locations"""
        loc = state[0]
        requests = self.get_poisson(loc)
        while requests == 0:
            requests = self.get_poisson(loc)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(0, (m-1)*m), requests)
        possible_actions_index.append((m-1)*m)
        random.shuffle(possible_actions_index)

        actions = [self.action_space[i] for i in possible_actions_index]

        return actions   

    def step_func(self, state, action, Time_matrix=Time_matrix):
        curr_loc = state[0]
        curr_hour = state[1]
        curr_day = state[2]
        
        pick_loc = action[0]
        drop_loc = action[1]
        
        pick_time = int(Time_matrix[curr_loc-1][pick_loc-1][curr_hour][curr_day])
        pick_hour = curr_hour + pick_time
        pick_day = curr_day
        
        if pick_hour > 23 :
            pick_hour = pick_hour % 24
            pick_day = (pick_day + 1) % 7

        trip_time = int(Time_matrix[pick_loc-1][drop_loc-1][pick_hour][pick_day])

        if action == (0,0): # driver does not accept ride for 1 hour
            reward = -1 * C
            next_loc = curr_loc
            next_hour = curr_hour + 1  
            next_day = curr_day
        else: # driver accepts ride
            reward = (R * trip_time) - (C * (pick_time + trip_time))
            next_loc = drop_loc
            if pick_hour == curr_hour & trip_time == 0:
                next_hour = curr_hour + 1
            else:
                next_hour = pick_hour + trip_time
            next_day = curr_day

        if next_hour > 23 :
            next_hour = next_hour % 24
            next_day = (next_day + 1) % 7
     
        next_state = (next_loc, next_hour, next_day)

        self.total_reward = self.total_reward + reward
        self.total_trips = self.total_trips + 1
        if (pick_time + trip_time) > 0:
            self.total_time = self.total_time + pick_time + trip_time
        else:
            self.total_time = self.total_time + 1

        is_terminal = False
        if self.total_time >= 720:
            is_terminal = True

        return next_state, reward, is_terminal
    
    def tracked_data(self):
        return self.total_reward, self.total_time, self.total_trips

    def reset(self):
        self.loc_space=np.random.choice(np.arange(1,m+1)) 
        self.hour=np.random.choice(np.arange(0,t)) 
        self.day=np.random.choice(np.arange(0,d)) 
        self.total_reward=0
        self.total_time=0
        self.total_trips=0
        self.state_init = (self.loc_space,self.hour,self.day)
        return self.state_init