# Data 

Services to check the quality of your data:
* Azure Purview
* Collibra
* Profisee
* Apache Griffin 
* Apache Atlas
* admunsen

Data Storage

 * PostgreSQL

<br>

## TileDB

[Data Economics](https://www.youtube.com/watch?v=aWWJlDA42aE)

 * Production: How the data is produced, and stored? 
 * Distribution: Who has access to the data? and by which channel? 
 * Consumption: How to consume the data (i.e., computation). 
     * Usually the focus 
     * But usually domain-specific 

Solution: 

 * Data in a universal analysis-ready format 
     * No ETL, no copies 
     * Unified governance 
     * Built-in marketplace
 * Universal data management platform 
     * One infrastructure, any backend, any scale 
     * Common for all data app

TileDB is likely be your solution
 * MultiDimensional arrays 
 * secure governance & collaboration 
 * Scalable, serverless compute 
 * ta and code sharing and monetization 
 * Pay-as-you-go, consumer pays 

## [Submodular Optimization for Minimizing Redundancy in Massive DataSets](https://www.youtube.com/watch?v=vJ3ErkmUpLU)

Increases in size mean increases in redundancy: 
* Duplicates 
    * in training sets can waste computational resources 
    * between training and test sets can cause overly optimistic estimates of performance 
    
$$
f(X \cup \{v\}) - f(X) \ge f(Y \cup \{v\}) - f(Y)
$$

where $X \in Y$ and $v \notin Y$

* Diminishing returns property: the gain from adding in some specific element $v$ to $X$ decreases, or stays the same, as other elements are added to $X$
* These functions don't require **continuous** or **differentiable**
* It's been shown that greedy algorithm will find a subset within a constant factor of $1-e^{-1}$ of the optimal subset, and empirical results show that the subset if almost always much closer


### Feature-based Function
A simple class of submodular function are feature-based functions 

$$
f(X) = \sum_{d=1}^D \phi ( \sum_{x \in X} x_d)
$$
where 
* $\mathbf{X}$ is the set of selected examples, 
* $x$ is an individual example 
* $\mathbf{D}$ is the number of dimensions 
* $d$ is the index of a particular dimension 
* $\phi$ is a concave function (e..g, sqrt or log)

### Graph-based Function
Facility location is another submodular function, and has been used to specify the location of new facilitate 

$$
f(X) = \sum_{y \in Y} \max_{x\in X} \phi (x,y)
$$
where 
* $\mathbf{Y}$ is the full set of items
* $\mathbf{X}$ is the set of selected items, 
* $\phi$ is a similarity function that returns the similarity between 2 examples (i.e., a graph) 
* $y, x$ are individual examples 

### Summary 
It's important to choose the right sumodular function
1. [Feature-based functions][Feature-based Function]hen each feature is a "quality of the data," they work well. A higher value means more of that quality in the data (i.e., word counts work well, pixel values do not)
2. [Graph-based Function] (e.g., facility location)can be used in a wide range of situations, but they need quadratic memory to store the similarity matrix, which takes up a lot of space.

`apricot` plays nice with `PyTorch` or `TensorFlow`
