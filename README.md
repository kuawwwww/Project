# Genetic Algorithm in solving TSP (GA)
> TSP is the NP-hard problem. This project uses a genetic algorithm to compute the optimal individuals in the last generation population, which in turn can be used as an approximate optimal solution to the problem. 

## Table of Contents
* [TSP problem](#tsp-problem)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Code Structure](#code-structure)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## TSP problem
TSP, or traveling salesman problem: Suppose a traveling salesman has to visit n cities, and he must choose the path to take, with the constraint that each city can only visit once, and finally return to the origin. The optimization goal of the problem is to minimize the path length.  

- Provide general information about your project here.
- What problem does it (intend to) solve?
- What is the purpose of your project?
- Why did you undertake it?
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


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
This project uses a genetic algorithm to solve the TSP problem:   

<div align=center><img src="https://github.com/kuawwwww/Project/blob/main/GA_plot.png" width="400"></div>

 
* First a permutation approach encodes the sequence of visited cities, which ensures that each city passes through and only once. 
* Then an initial population is generated and the distance of all cities is traversed to calculate the fitness function. 
* Then selection is performed with Roulette wheel selection, using Partially-matched crossover and simple mutation to determine the crossover operator and variation operator.  

To enhance robustness, we store the cities in [cn.csv](https://github.com/kuawwwww/Project/blob/main/cn.csv "悬停显示"). So you can update the table directly when you need to use it.  But remember, the 1111111


## Code Structure
Only part of the important code is presented below, the full code can be seen at:  [TSP(GA).py](https://github.com/kuawwwww/Project/blob/main/cn.csv "悬停显示")  
Main content：  
* GA process
* Visualization

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
                path_a[x[x != i]] = path_a2[i] #由于x是array
            if len(y) == 2:
                path_b[y[y != i]] = path_b2[i]
        return path_a, path_b
    
    def Cross(self):

        for i in range(0,self.select_num,2): #步长为2的原因是因为要取i和i+1进行交叉
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

    data = pd.read_csv("D:\python and optimization\cn.csv")
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

## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- 
- Improvement to be done 2


## Acknowledgements
Give credit here.
- This project was inspired by Tenglong Hong and Yucong Shi.
- This project was based on [this tutorial](https://www.example.com).



