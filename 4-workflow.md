# Workflow

Workflow management: 
* https://github.com/kubeflow/pipelines/
* https://github.com/couler-proj/couler
* https://github.com/argoproj/argo-workflows/tree/master/sdks/python



Reality of AI workloads
* Data ingestion 
* Concept drift monitoring 
* Data Drift monitoring 
* Feature storage 
* Retraining 
* Workload upgrades 
* Building a model 
* Resource scaling 
* Security 
* Model versioning 
* Pipeline Monitoring 
* A/B testing traffic routing
* Model API serving 
* Accuracy Tracking 
* Maintenance 

Programs for enterprise ML platform
* Snowflake (all-in-one) to store both structured and unstructured data
* SaturnCloud: data science cloud environment. 


ML Life Cycle 
1. Raw data 
    * File 
	* Batch 
	* Streaming
2. Data Prep 
	* Data preprocessing
	* Feature engineering 
	* Data transformation
3. Training 
	* Multiple algorithm s
	* Hyperparameter 
	* Model comparison 
	* Model evaluation
4. Deploy
	* Integration
	* Monitoring



Argo Project : a a set of Kubernetes-native tools for deploying and running apps, managing clusters, and do GitOps right 
* Argo Workflows: Kubernetes-native workflow engine 
* Argo Events: Event-based dependency management for Kubernetes 
* Argo CD: Declarative continuous delivery with a fully-loaded UI
* Argo Rollouts: Advanced K8s progressive deployment strategies 

Argo Workflows 
* The container-native workflow engine for Kubernetes 
	* ML pipelines 
	* Data processing
	* Infrastructure automation
	* continuous delivery/Integration
* CRDS and Controllers
	* Kubernetes custom resources that natively integrates with other K8s resources (volumes, secrets, etc.)
* Interfaces
	* CLI: manage workflows and perform operations (submit, suspend, delete/etc.)
	* Server: REST & GRPC interfaces
	* UI: manage and visualize workflows, artifacts, logs, resource usages analytic, etc.
	* Python and Java SDKS
    
## Agile Data Science

 * Agile is a work management technique that uses time-limited iterations to complete tasks.
 * Sprint: 1-4 week cycle delivering working increment (i.e., output that can be tested). 
 * Story: single unit of work completed within a sprint 
 * Epic: collection of related stories 
 
 

## Hydra & hydrazen

 * Standardize the process of designing your project 
 * Make your project configurable 
     * Configure deeply-nested parameters 
     * Change algorithms and models robustly 
 * Make your code reproducible (leave breadcrumb)
 * Enable **scalable** workflow

Can be used for creating games 

## [Ray](https://www.youtube.com/watch?v=wl4tvru9_Cg&ab_channel=PyData)

Overall: Kubeflow or Apache Airflow
 1. Preprocess: Spark or Dask 
 2. Checkpoint: HDF5, S3
 3. Train: XGBoost

Challenges: 
 * Performance overheads
     * Serialization/Deserialization 
     * data materialized to external storage 
 * Implementation/Operinal compelxity 
     * Croos-lang, cross-workload
     * CPUs vs. GPUs
 * Misisng opreations
     * Per-epoch shuffling

Why Ray?

 * Efficient data layer 
     * Zero-copy reads, shared -memory ojbect sotre 
     * Locality-aware cheduling 
     * Object transfer protocosl 
 * General purpose 
     * Resoruce-based scheudling 
     * Higly scalable 
     * Robust primtives 
     * Easy to program distirubted programs 
     
Ray Datasets (not a dataframe library)
1. Universal Data loading 
2. Last Mile Preprocessing 
3. Parallel GPU/CPU Compute

Universal Data Loader 
* HDF5 
* S3
* Spark
* Dask
* Modin

powered by Apache Arrow 

```{python}
ds = ray.data.read_csv("") 

```

## [Snowflake and Tecton](https://www.youtube.com/watch?v=tETMa7xStr4&ab_channel=PyData)

production ML pipelines must: 
* Transform raw data from batch, streaming (e.g., Kafka), and real-time (RPC) sources
* Serve data with point-in-time correctness for model training 
* Serve data at low latency and high concurrency for model serving 
* Ensure model training/serving parity 
* Backfill streaming and real-time features 


Snowflake and Tecton can fix these roadblocks. 