To store and access big data on cloud, we can use `ReferenceFileSystem` (a virtual implementation for ffspec)
We then can use only zarr to access the data in the form of HDF5, TIFF or grib2. 
For documentation: [check](https://github.com/fsspec/kerchunk)
For example, [check](https://nbviewer.org/gist/rsignell-usgs/02da7d9257b4b26d84d053be1af2ceeb)