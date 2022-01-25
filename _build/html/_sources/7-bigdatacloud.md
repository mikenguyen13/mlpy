# Big Data
To store and access big data on cloud, we can use `ReferenceFileSystem` (a virtual implementation for ffspec)
We then can use only zarr to access the data in the form of HDF5, TIFF or grib2. 
For documentation: [check](https://github.com/fsspec/kerchunk)
For example, [check](https://nbviewer.org/gist/rsignell-usgs/02da7d9257b4b26d84d053be1af2ceeb)


# High Performance Python

* numba instead of numpy (just in time)  using C++ and Fortran 
* rapid instead of pandas
	* pandas -> cudf
	* numpy -> cupy
	* scikit-learn -> cuml
* Memory error: Read book High performance Python: practical performance programming for humans 
	* Use DASK (recommended), also Pysparak is similar. will be compatible with `afar` (Dask cluster)



Deploy Rapids Client 
* Local machine (with GPU)
* AWS Sagemaker
* Google Cloud vertex AI
* Azure Machine Learning 

Deploy Dask Cluster 
* Manual Installation 
* Dask-cloudprovider
* Dask Kubernetes
* Coiled
* Saturn Cloud
* Google Cloud Dataproc


