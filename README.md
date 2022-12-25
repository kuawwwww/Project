# Solving TSP via Genetic Algorithm(GA) and Q-Learning(QL)
> TSP is the NP-hard problem. This project uses genetic algorithm and Q-Learning to compute the optimal choices of orders for visiting cities, which in turn can be used as an approximate optimal solution to the problem. And this project makes a comparison between these two algorithms.

## Table of Contents
* [TSP problem](#tsp-problem)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Code Structure for GA](#code-structure-for-ga)
* [Code Structure for QL](#code-structure-for-ql)
* [Comparison for GA and QL](#comparison-for-ga-and-ql)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)



## TSP problem
TSP, or traveling salesman problem: Suppose a traveling salesman has to visit n cities, and he must choose the path to take, with the constraint that each city can only visit once, and finally return to the origin. The optimization goal of the problem is to minimize the path length.  


## Prerequisites
* python 3
* Windows or MacOS 
 
You'll also need to use a third library: numpy, pandas, matplotlib, and time. You can install them (which are themself pip packages) via
```python
pip install numpy
pip install pandas
pip install matplotlib
pip install time
```

## Usage
This project first uses a genetic algorithm to solve the TSP problem:   

<div align=center><img src="https://github.com/kuawwwww/Project/blob/main/GA_plot.png" width="400"></div>

 
* First a permutation approach encodes the sequence of visited cities, which ensures that each city passes through and only once. 
* Then an initial population is generated and the distance of all cities is traversed to calculate the fitness function. 
* Then selection is performed with Roulette wheel selection, using Partially-matched crossover and simple mutation to determine the crossover operator and variation operator.  

To enhance robustness, we store the cities in [cn.csv](https://github.com/kuawwwww/Project/blob/main/cn.csv "悬停显示"). So you can update the table directly when you need to use it.  But remember, you can download the file and make some changes. However, 'capital','lat' and 'lng' must be included as header.


## Code Structure for GA
Only part of the important code is presented below, the full code can be seen at:  [TSP(GA).py](https://github.com/kuawwwww/Project/blob/main/TSP(GA).py "悬停显示")  


Main content：  
* GA process
* Visualization
* Read csv

### Variable names: 
|**Variable names:**|**content**|
|:--:|:--:|
|self.maximize_interation|Maximum literation|
|self.population_size|population size|
|self.cross_prob|crossover probility|
|self.mutation_prob|mutation probility|
|self.select_prob|selection probility|
|self.data|Coordinate data of the city|
|self.num|number of cities|
|self.select_num|Determine the number of choices of offspring by selection probability|
|self.parent，self.child|Initialize parent and child|

### GA process:
1. Calculate the distance between cities (the code uses Euclidean distance as an example)

```python
    #computing the distance matrix
    def computing_distance(self): 
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res
```
2. Calculate the path length and output the initial solution

 ```python
 # Generate parents
    def generate_parents(self):
        sequence = np.array(range(self.num))
        for i in range(self.population_size):
            np.random.shuffle(sequence)
            self.parent[i,:] = sequence
            self.fitness[i] = self.compare_fitness(sequence) #Judgemnet of fitness
    
    # Evaluation function
    def compare_fitness(self,path):
        total_distance = 0
        for i in range(self.num - 1):   
            total_distance += self.distance_matrix[path[i],path[i+1]]
        total_distance += self.distance_matrix[path[-1],path[0]]
        return total_distance

    # Reveal the path 
    def out_path(self,path):

        res = str(path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(path[i] + 1) + '-->'
        res += str(path[0] + 1) + '\n'
        print(res)
        route_city = str(self.city_dict[path[0]]) + '-->'
        for i in range(1, self.num):
            route_city += str(self.city_dict[path[i]]) + '-->'
        route_city += str(self.city_dict[path[0]]) + '\n'
        print(route_city)
  ```
3. Selection: *Roulette wheel selection*  
Since the TSP problem is a minimization problem, the fitness function needs to be changed to maximize it. 

Fitness funtion:   
$$f'(x)=\ \frac{1}{f(x)}\ $$

Also, the distance is long, resulting in small differences between the fitness functions, so at this point, to avoid falling into prematureness, the selection probabilities we convert to:    

$$p=\ \frac{p_i}{\sum_{i} p_i}\ $$


```python
    # Generate childs
    def Select(self,index):
        pick = []
        rand_list = [np.random.uniform(0, 1 / min(self.fitness)) for i in range(self.select_num)]
        fit = 1 / self.fitness      
        cumsum_fit = np.cumsum(fit)         
        sumsum_fit = sum(cumsum_fit)
        for p in range(self.population_size):
            pick_pro = cumsum_fit[p] / sumsum_fit
            pick.append(pick_pro)
        i, j  = 0, 0 
        index = []
        while i < self.population_size and j < self.select_num: #Select candidate from parents
            if pick[i] >= rand_list[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.child = self.parent[index,:] #Generate child
  ```


4. Crossover: *PMX*   
We choose **partial matching crossover**. 
> The reason is that it ensures that genes appear only once in each chromosome and no duplicate genes appear in a chromosome.    
```python
    #  Partially-matched crossover, PMX , the main part in GA       
    def intercross(self,path_a,path_b):       
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r1 == r2:
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        path_a1 = path_a.copy()
        path_b1 = path_b.copy()
        for i in range(left,right+1):
            path_a2 = path_a.copy()
            path_b2 = path_b.copy()
            path_a[i] = path_b1[i]
            path_b[i] = path_a1[i]
            x = np.argwhere(path_a == path_a[i]) 
            y = np.argwhere(path_b == path_b[i])
            if len(x) == 2:
                path_a[x[x != i]] = path_a2[i] 
            if len(y) == 2:
                path_b[y[y != i]] = path_b2[i]
        return path_a, path_b
    
    def Cross(self):

        for i in range(0,self.select_num,2): 
            if self.cross_prob >= np.random.rand():
                self.child[i, :], self.child[i + 1, :] = self.intercross(self.child[i, :],self.child[i + 1, :])
  ```
  
 5. Mutation and reverse  
 > Simple mutation: Ensures that the DNA is diverse and that the algorithm can explore possible solutions in a sufficient number of directions.   
 > Reverse: The approximation reversal, which means that the next DNA does not go to the next generation if it is not better, is also a part of the selection. To reduce data redundancy, we are not replacing all at once  
  ```python
    # Mutation function, change part of DNA of childs with a rather low probability
    def Mutation(self):

        for i in range(self.select_num):
            if np.random.rand() <= self.mutation_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r1 == r2:
                    r2 = np.random.randint(self.num)
                self.child[i, [r1, r2]] = self.child[i, [r2, r1]]
    
    # Reverse function, to reverse part of DNA of the childs
    def Reverse(self):

        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r1 == r2:
                r2 = np.random.randint(self.num)
            left, right = min(r1, r2), max(r1, r2)
            reverse = self.child[i,:].copy()
            reverse[left:right + 1] = self.child[i,left:right+1][::-1]
            if self.compare_fitness(reverse) < self.compare_fitness(self.child[i,:]):
                self.child[i,:] = reverse
```   

### Main function
``` python   
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
```
## Code Structure for QL
Only part of the important code is presented below, the full code can be seen at:  [TSP(QL).py](https://github.com/kuawwwww/Project/blob/main/TSP(GA).py "悬停显示") 

Main content：  
* QL process
* Visualization
* Read csv

### Variable names: 
|**Variable names:**|**content**|
|:--:|:--:|
|self.epsilon|If the random number is larger than the parameter, take random action|
|self.gamma|Parameter gamma in the update equation|
|self.lr|Parameter alpha in the update equation|
|self.data|Coordinate data of the city|
|self.city_dict|Record the names of the cities|
|self.num|number of cities|
|self.R_table|Reward function table|
|self.Q_table|Q_table for Q-Learning|
|self.space|Set of nodes(cities)|
|self.iterate_results|Record the training results|

### QL process:
1. Calculate the distance between cities (the code uses Euclidean distance as an example) and meanwhile, initialize the Reward table using the distances

```python
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
```
2. Train the Q_table
> We use Q-Learning to solve the TSP. Therefore, we need to construct Q-Table and then update it.
> We choose the **Greedy Algorithm** to initialize the Q-Table, but we also consider random factors to take generate next action.
```python
    def Train(self):
        iterate_results = []  #Store each training's result
        for i in range(500):  #In general,we train the model for 50000 times
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
 ```
 3. Update Q_table using the following euqation
 > Update equation:
 <div align=center><img src="https://github.com/kuawwwww/Project/blob/main/QL_Equation_plot.png" width="600"></div>
 
 ```python
                # Update Q_table through the above results
                if j != int(self.num/2):
                    self.Q_table[s, a] = (1 - self.lr) * self.Q_table[s, a] + self.lr * (self.R_table[s, a] + self.gamma * max_value)
                else:
                    self.Q_table[s, a] = (1 - self.lr) * self.Q_table[s, a] + self.lr * self.R_table[s, a]
                path.append(a)
                self.Q_table[a, 0] = (1 - self.lr) * self.Q_table[a, 0] + self.lr * self.R_table[a, 0]
            # End position
            path.append(0)   #We should go back to the origin at the end
 ```
 4. Test the training result
 ```python
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
```
5. Get the optimal route w.p.t the Q_table being trained before
 ```python
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
```
### Main function
``` python  
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
```
## Comparison for GA and QL
1. GA is an exact solution, but has the risk of local optimum; QL is an approximate solution, but is less stable.  
2. GA is an interaction in a separate instance of the environment, so there is no feature of temporal decision making like QL. It also ignores the fact that policy is actually a mapping from state to action, and no features are learned from the interaction with the environment. Therefore, QL is generally more effective in finding the right policy.  
3. There is no value function in GA, and there is no dynamic learning process in the life cycle of each agent, so only problems where the policy space is small enough or easily structured are suitable for solving with genetic algorithms. However, when the agent cannot perceive the environment well, evolutionary algorithms, for example, are more advantageous than reinforcement learning.  

## Room for Improvement

Room for improvement:
- At present, GA code is prone to falling into precocity because the difference in fitness function is small, which is also one of the drawbacks of GA.
- For static problems like TSP, reinforcement learning still performs worse than GA, however, there is still a promising prospect to combine Reinforcement learning with TSP and other NP hard problems. 
- To enhance robustness, Read.csv wrapping can be performed at a further stage.


## Acknowledgements
Give credit here.
- This project was inspired by Tenglong Hong and Yucong Shi.
- Yucong Shi finished the coding for GA
- Tenglong Hong finished the coding for QL
- This project was based on *Algorithms for optimization by Mykel J. Kochenferfer*.




