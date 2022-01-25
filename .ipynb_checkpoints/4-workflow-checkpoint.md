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

