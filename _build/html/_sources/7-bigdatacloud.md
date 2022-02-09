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


## Amazon Web Services 

When working with big data that are stored online, we can utilize AWS to **store**, and **process** data. 

To work with AWS, we have to follow these steps: 

1. [Create a AWS account]
2. [Create Cluster EMR]
3. [Data Storage Amazon S3]
4. [Specify PUTTY]
5. [Web Interface WinSCP]

### Create a AWS account

Create a AWS account with your credit card (Amazon use the pay-as-you-go service charge)

### Create Cluster EMR

1. Under `Services` tab on the top left, select `Analytics`, then `EMR`
2. Choose `Cluster on EC2`
3. Create a new cluster (the default setting you get you going)
4. If you've already a cluster before, and want all of the previous setting, mark the clusters you want to clone, then hit "Clone"

Recommendation: Setup `auto-terminate`  so that you won't be overcharged. 

### Data Storage Amazon S3

1. Under `Services` tab, scroll all the way down to `Storage`, pick `S3` Scalable storage in the Cloud). 
2. In the `Buckets` tab, hit `Create bucket` (A bucket on the cloud is equivalent to a drive on your local machine). 
3. Inside a bucket, you can start creating folders just like you do on your local machine.
4. In side the folder, under `Properties`, the "S3 URI" is your folder directory. 

### Specify PuTTY

1. [Download PuTTY](https://www.putty.org/)
2. [Set up key with AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html)
3. After having your private key file for authentication in the `.ppk` format, you can go back to your clusters on `EMR on EC2`. 
4. Under "Summary" tab, click `Connect to the Master Node Using SSH`, there you will see instructions for Windows or Mac/Linux. 
5. For Windows, copy the Host Name field in the form of username@... (e.g., hadoop@ec2-...)
6. Go back to PuTTY, paste the Host Name field to the "Host Name (or IP address)" box under Session 
7. Expand `SSH`, then expand `Auth`, browse your private key file for authentication. If you see EMR, then you are there. 

### Web Interface WinSCP

To move your files (data, script) to AWS, you can use WinSCP 

1. [Download WinSCP ](https://winscp.net/eng/download.php)
2. Click `New Session` close to top left 
3. Host name is the long character after the @ sign on your EMR Clusters
4. User name is the character before the @ sign on your EMR Clusters 
5. Password is the private key you have. Click `Advanced` -> `SSH` -> `Authentication` -> `...` to browse you key. Click OK. 
6. Click `Login`. Voila! You're there.



