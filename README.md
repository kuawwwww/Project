# Genetic Algorithm in solving TSP (GA)
> TSP is the NP-hard problem. This project uses a genetic algorithm to compute the optimal individuals in the last generation population, which in turn can be used as an approximate optimal solution to the problem. 

## Table of Contents
* [General Info](#general-information)
* [Prerequisites](#prerequisites)
* [Screenshots](#screenshots)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
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


## Screenshots
<img src="https://github.com/kuawwwww/Project/blob/main/GA_plot.png" width="400">

## Usage
To enhance robustness, we store the cities in [我的博客](http://blog.csdn.net/guodongxiaren "悬停显示"). So you can update the table directly when you need to use it.  

This project uses a genetic algorithm to solve the TSP problem: first a permutation approach encodes the sequence of visited cities, which ensures that each city passes through and only once. Then an initial population is generated and the distance of all cities is traversed to calculate the fitness function. Then selection is performed with Roulette wheel selection, using Partially-matched crossover and simple mutation to determine the crossover operator and variation operator.
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`


### Project Structure
Project is:  _complete_ 


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
