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
Project is:  _complete_ 

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
2. Selection: PMX 
We choose **partial matching crossover**. 
> The reason is that it ensures that genes appear only once in each chromosome and no duplicate genes appear in a chromosome.  
PMX determines the crossover region by randomly selecting two crossover points. After performing the crossover, we usually get two invalid chromosomes and individual genes will be duplicated. In order to repair the chromosomes, we can establish a matching relationship for each chromosome within the crossover region and then apply this matching relationship to the duplicated genes outside the crossover region to eliminate the conflict.


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
            x = np.argwhere(path_a == path_a[i]) #返回非0的数组索引位置，索引数组条件为path_a == path_a[i]
            y = np.argwhere(path_b == path_b[i])
            if len(x) == 2:#当DNA中有两个位置是同一样的数，换成
                path_a[x[x != i]] = path_a2[i] #由于x是array
            if len(y) == 2:
                path_b[y[y != i]] = path_b2[i]
        return path_a, path_b
    
    def Cross(self):

        for i in range(0,self.select_num,2): #步长为2的原因是因为要取i和i+1进行交叉
            if self.cross_prob >= np.random.rand():
                self.child[i, :], self.child[i + 1, :] = self.intercross(self.child[i, :],self.child[i + 1, :])

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
    
    # Born function, to replace the original parents with newborn childs
    def Born(self):
        index = np.argsort(self.fitness)[::-1]
        self.parent[index[:self.select_num],:] = self.child


# Main function
def Main(data,city_dict):

    TSP_GA = Genatic_TSP(data,city_dict)

    TSP_GA.generate_parents()

    for i in range(TSP_GA.maximize_interation): 
        TSP_GA.Select(i)    #Selction
        TSP_GA.Cross()      #Crossing
        TSP_GA.Mutation()   #Mutation
        TSP_GA.Reverse()    #Reverse
        TSP_GA.Born()       #Replace Parents

        for j in range(TSP_GA.population_size):
            TSP_GA.fitness[j] = TSP_GA.compare_fitness(TSP_GA.parent[j,:])
        index = TSP_GA.fitness.argmin()
        if (i + 1) % 50 == 0:
            print('The shortest distance after ' + str(i + 1) + ' step is : ' + str(TSP_GA.fitness[index]))
            print('The optimal path after ' + str(i + 1) + ' step is : ')
            TSP_GA.out_path(TSP_GA.parent[index, :])  # Reveal the path 
        TSP_GA.best_fit.append(TSP_GA.fitness[index])
        TSP_GA.best_path.append(TSP_GA.parent[index,:])

    # Draw the trend of interating results
    fig, ax = plt.subplots()
    ax.plot(TSP_GA.best_fit)
    plt.title("The trend of the result", fontsize=10)
    plt.xlabel("Iterating times", fontsize=10)
    plt.ylabel("Distance", fontsize=10)
    plt.savefig('GA_trend.png', dpi=300)
    plt.show()

    # Draw the optimal route
    fig, ax = plt.subplots()
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(y, x, linewidths=0.1)
    for i, txt in enumerate(range(1, len(data) + 1)):
        ax.annotate(TSP_GA.city_dict[txt-1], (y[i],x[i]))
    res_ = TSP_GA.best_path[0]
    x_ = x[res_]
    y_ = y[res_]
 

    for i in range(len(data) - 1):
        plt.quiver(y_[i],x_[i],y_[i + 1] - y_[i],x_[i + 1] - x_[i], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(y_[-1], x_[-1], y_[0] - y_[-1],x_[0] - x_[-1], color='r', width=0.005, angles='xy', scale=1,
               scale_units='xy')
    plt.savefig('GA.png', dpi=300) #保存图片，由于直接输出的清晰度不高
    plt.show()      
    return TSP_GA


# Main function    
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


start =time.clock()
end = time.clock()
print('Running time: %s Seconds'%(end-start))

```

## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
