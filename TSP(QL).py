# Q-Learning TSP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class Qlearning_TSP(object):
    def __init__(self, data, city_dict, epsilon = 0.8,gamma = 0.1,lr = 0.1,iterate_times = 50000):
        self.epsilon = epsilon      # Parameter, if the random number is larger than it, take random action
        self.gamma = gamma          # Parameter in the equation that updates Q_table
        self.lr = lr                # Parameter in the equation that updates Q_table
        self.data = data
        self.city_dict = city_dict
        self.num = len(data)   # number of cities
        self.distance_matrix, self.R_table = self.computing_distance()
        self.Q_table = np.zeros((self.num, self.num))   #Initialize Q table
        self.space = [i for i in range(self.num)]       #Set of nodes 
        self.iterate_times = iterate_times
        self.iterate_results = self.Train()             #Store the training results


    def computing_distance(self): #computing the distance matrix
        res = np.zeros((self.num, self.num))
        k = 0
        for i in range(self.num):
            for j in range(i + 1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
                if res[i,j] >= k:
                    k = res[i,j]
        R_table = k - res  # Initialize R_table (Reward table, We initialize ench enty with the difference between its original distance with maximum distance)
        return res, R_table

    # Train the Q_table and meanwhile test the training result
    def Train(self):
        iterate_results = []  #Store each training's result
        for i in range(self.iterate_times):  #In general,we train the model for 50000 times
            # Initial position
            path = [0]
            # Each round, we obtain (num-1) positions
            for j in range(self.num - 1):
                s = path[j]  # Current position
                s_row = self.Q_table[s]  # matching the current position with the row in Q_table
                remaining = set(self.space) - set(path)  # remaining nodes

                # Find the maximal value in remainging nodes
                max_value = -1000  
                a = 0

                # We use greedy method to obtain next action
                for rm in remaining:
                    Q = self.Q_table[s, rm]
                    if Q > max_value:
                        max_value = Q
                        a = rm

                # Randomize next action to avoid trapping in local optimum
                if np.random.uniform() > self.epsilon:
                    a = np.random.choice(np.array(list(set(self.space) - set(path))))

                # Update Q_table through the above results
                if j != int(self.num/2):
                    self.Q_table[s, a] = (1 - self.lr) * self.Q_table[s, a] + self.lr * (self.R_table[s, a] + self.gamma * max_value)
                else:
                    self.Q_table[s, a] = (1 - self.lr) * self.Q_table[s, a] + self.lr * self.R_table[s, a]
                path.append(a)
                self.Q_table[a, 0] = (1 - self.lr) * self.Q_table[a, 0] + self.lr * self.R_table[a, 0]
            # End position
            path.append(0)   #We should go back to the origin at the end

            # Test the train result (Obtain shortest path w.p.t current Q_table)
            result = [0]
            for _ in range(self.num-1):
                loc = result[-1]
                new_remaining = set(self.space) - set(result)  # remaining nodes

                # Find the maximal value in remaining node
                new_max_value = -1000 
                a = 0

                # We repeat the action similar to update Q_table before
                for rm in new_remaining:
                    new_Q = self.Q_table[loc, rm]
                    if new_Q > new_max_value:
                        a = rm
                        new_max_value = new_Q
                result.append(a)
            result.append(0) #We need to go bcak to the origin

            # Compute the current total distacne 
            length = 0
            for v in range(1, self.num):
                length += self.distance_matrix[result[v - 1], result[v]]

            # Print the result with an interval of 50 rounds
            if (i + 1) % 50 == 0:
                print(f"Start {i+1}'s round of training")
                print(f"The shortest distance after {i+1}'s round is {length}"+'\n')
            iterate_results.append(length)
        return iterate_results

    # Get the optimal route w.p.t the Q_table being trained before
    def get_route(self):

        # The following steps are similar to the part in training before 
        result = [0]
        for i in range(self.num):
            loc = result[-1]
            remaining = set(self.space) - set(result)  #remaining nodes
            max_value = -1000
            # Find the maximal value in remaining node
            a = 0
            # We use greedy method to choose next action
            for rm in remaining:
                new_Q = self.Q_table[loc, rm]
                if new_Q > max_value:
                    a = rm
                    max_value = new_Q
            result.append(a)
        result.append(0)
        length = 0
        for v in range(1, self.num):
            length += self.distance_matrix[result[v - 1], result[v]]
        result = [i+1 for i in result[:-1]]

        route = str(result[0]) + '-->'
        for i in range(1, self.num):
            route += str(result[i] ) + '-->'
        route += str(result[0]) + '\n'

        route_city = str(self.city_dict[result[0]-1]) + '-->'
        for i in range(1, self.num):
            route_city += str(self.city_dict[result[i]-1]) + '-->'
        route_city += str(self.city_dict[result[0]-1]) + '\n'

 
        print(f"The shortest distance after 50000 times training is：{length}"+'\n')
        print(f"The optimal path after 50000 times training is: ")
        print(route)
        print(route_city)

        out_result = np.array([0]*self.num)
        for i in range(self.num):
            out_result[i] = result[i]
        out_result -= 1

        return out_result

# Main function 
def Main(data,city_dict):

    TSP_QL = Qlearning_TSP(data,city_dict)
    TSP_QL.Train()
    res_ = TSP_QL.get_route()

    # Draw the trend of training results
    fig, ax = plt.subplots()
    ax.plot(TSP_QL.iterate_results)
    plt.title("The trend of the result", fontsize=10)
    plt.xlabel("Iterating times", fontsize=10)
    plt.ylabel("Distance", fontsize=10)
    plt.show()

    # Draw the optimal route
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    #plt.figure(figsize=(40,20))
    ax.scatter(y, x, linewidths=0.1)
    for i, txt in enumerate(range(1, len(data) + 1)):
        ax.annotate(TSP_QL.city_dict[txt-1], (y[i],x[i]))
    x_ = x[res_]
    y_ = y[res_]
    for i in range(len(data) - 1):
        plt.quiver(y_[i],x_[i],y_[i + 1] - y_[i],x_[i + 1] - x_[i], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(y_[-1], x_[-1], y_[0] - y_[-1],x_[0] - x_[-1], color='r', width=0.005, angles='xy', scale=1,
               scale_units='xy')
    
    plt.show()

    return TSP_QL

# Main function    
if __name__ == '__main__':

    data = pd.read_csv("cn.csv")
    data = data[(data['capital'] == 'admin') | (data['capital'] == 'primary')]
    cities = data['city'].values
    loc_x_vals = data['lat'].values
    loc_y_vals = data['lng'].values
    n = data.shape[0]
    city_dict = dict(zip([i for i in range(0,n)],cities))
    data = np.array([[loc_x_vals[i],loc_y_vals[i]] for i in range(0,n)])

    np.random.seed(1)
    Main(data,city_dict)

# start = time.clock()
# end = time.clock()
# print('Running time: %s Seconds'%(end-start))

