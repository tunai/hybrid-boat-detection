This guide reflects the installation steps of Detectron2 on Windows 10 for my particular hardware. Note that for different GPUs, operational systems, etc. the layout/steps could be slightly different.

## Pre-requisites: 
* [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) with C++ Build Tools. This tutorial uses the *Community* version. 
* CUDA > 10.1 (once installed, run ```nvcc --version``` on windows cmd to check version and installation completion)
* [Anaconda](https://www.anaconda.com/products/individual)

## Steps: 

1. Run anaconda prompt, create an environment, activate it and install pip: 
```
conda create environment --n myenv python=3.7
conda activate myenv
conda install pip
```
Note: make sure you create an environment with Python 3.7. The installation does not work with Python>3.7.

2. Install pytorch 1.3.1 and torchivsion 1.2.3:
```
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch 
```
Note: installing newer versions of pytorch/torchvision (e.g., torchvision>0.5) will hinder the installation. 

3. Install opencv:
```
conda install -c conda-forge opencv
```

4. Install fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
Success message: "Successfully installed fvcore-0.1.2 portalocker-2.0.0 pywin32-228 pyyaml-5.3.1 tabulate-0.8.7 termcolor-1.1.0 tqdm-4.51.0 yacs-0.1.8"

5. Install cython
```
conda install -c anaconda cython
```
6. Install picocotools:
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
Success message: "Successfully installed pycocotools-2.0"

7. Change the content of two files manually:

File 1: 
  {anaconda3 path}\pkgs\pytorch-1.3.1-py3.7_cuda101_cudnn7_0\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
```
static constexpr size_t DEPTH_LIMIT = 128; 
      change to -->
static const size_t DEPTH_LIMIT = 128;
```
Note: for different versions of pytorch, simply search for "DEPTH_LIMIT" and change "constexpr" for "const".

File 2: 
  {anaconda3 path}\pkgs\pytorch-1.3.1-py3.7_cuda101_cudnn7_0\Lib\site-packages\torch\include\pybind11\cast.h
```
explicit operator type&() { return *(this->value); }
      change to -->
explicit operator type&() { return *((type*)this->value); }
```
    
8. Run VS2019 bat file.
(still on anaconda prompt with your environment activated):
```
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 
```
Note: the path for VS2019 *Enterprise* is going to be different.

9. Download and install [conansherry's Detectron2](github.com/conansherry/detectron2):
```
git clone https://github.com/conansherry/detectron2
cd detectron2
python setup.py build develop
```
 
10. Test detectron2: 
```
python
import detectron2
detectron2.__version__
```
Success output message: "0.1"



