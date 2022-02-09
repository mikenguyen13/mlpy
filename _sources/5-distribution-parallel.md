# Distributed 
Distributed ML Pipelines (https://www.youtube.com/watch?v=dNzb_-JD6T0)
* Data preprocessing 
	* feature engineering
* Distributed model training 
	* Hyperparameter tuning
	* Model selection/architecture search 
	* Distribute training strategies (PS and all reduce)
	* Scheduling techniques (priority, gang, elastic scheduling, etc.)
* Model serving 
	* Replicated services 
	* Sharded services 
	* Event-driven processing
* Workflow orchestration 

## Dask Using Saturn Cloud [https://www.youtube.com/watch?v=G303aRAwsoE]

Parallel computing 
 * Python is friendly but not fast 
 * cant utilize multithreading (due to Global Interpreter Lock)
 * enable distribution of work across multiple cores and machines
 * allows us to 
     * train models faster 
     * tune hyper parameters
     * feature engineer 

Dask 
 * free 
 * in python 
 * great computing speed


User Interfaces 
* High-level: scalable venison of NumPy, Pandas
    * same syntax, but computation is faster
* Low level: parallelized custom algorithm 

Dask Terminologies
 * Dask Dataframe is a collection of smaller pandas dataframes that are distributed across your cluster (Dask Dataframe only reads smaller chunk of lines). 
 * Dask Array: mimic the functionality of NumPy arrays using a distributed backend. 
 * Dask Bags: 
     * Used for taking a collection of Python objects and processing them in parallel 
     * Parallelized simple computations on unstructured or semi-structured data
     * then handing off to Dask Dataframes 


