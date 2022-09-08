# HTTO stands for High Throughput Tomography pipeline
* A Python tool for parallel read of h5 tomographic data using MPI protocols
* The data can be re-chunked, saved and re-loaded (e.g. projection or sinogram-wise)
* The data is then can be processed by any tomographic packages, e.g. TomoPy, ASTRA

### Installation of software prerequisites:
* Copy the repository from Guthub with `git clone git@github.com:dkazanc/HTTO.git`
* Due to a significant number of dependencies such as: CuPy, PyTorch, Cudatoolkit, etc., it is easier to conda install an environment from the explicit [list](https://github.com/dkazanc/HTTO/blob/master/conda/specs_env_explicit.txt) file as: 
```
conda create --name htto --file conda/specs_env_explicit.txt
```
* This will create the **htto** codna environment which you can activate with `conda activate htto`

### Running the code:
