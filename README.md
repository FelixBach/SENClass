# SENClass - Processing of Sentinel-1 data

## Installation 

The easiest way to install the package is via a Conda environment. To do so follow these steps:

Download code as zip and unzip the folder
start anaconda prompt
Change in the anaconda prompt the directory to the unziped folder with .yml-file.
After that you can copy these commands in the anaconda prompt. You have only to adjust the path to the notebook
```
conda env create -f env_geo.yml
```
```
conda activate python_senclass
```
```
pip install git+https://github.com/FelixBach/SENClass.git```
```
```
pip install jupyter notebook
```
```
python -m ipykernel install --user --name python_senclass --display-name "senclass-env"
```
```
cd “path/with/notebook”
```
```
jupyter trust SENClass.ipynb
```
```
jupyter notebook SENClass.ipynb
```  
After the juypter notebook boots, you may still need to select the correct kernel (senclass-env).

Please note that you have to install GDAL and rasterio manually, if you don not use anaconda.
To do this please download the binaries (.whl-files) from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

Alternatively, the package can be installed without Anaconda using the following line. 
```
pip install git+https://github.com/FelixBach/SENClass.git
```
In the folder ipynb you can call main.py to test the functionality of the program. 


- If there are problems with the package, please open an [issue](https://github.com/FelixBach/GEO419/issues)
