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
Success output message: "0.1".

You're done, congrats! For reference, this is how my "requirements.txt" file looked at this point: 
```
absl-py==0.11.0
cachetools==4.1.1
certifi==2020.6.20
cffi @ file:///C:/ci/cffi_1600699250966/work
chardet==3.0.4
cloudpickle==1.6.0
cycler==0.10.0
Cython @ file:///C:/ci/cython_1594834055134/work
-e git+https://github.com/conansherry/detectron2@72c935d9aad8935406b1038af408aa06077d950a#egg=detectron2
fvcore @ git+https://github.com/facebookresearch/fvcore@f6909f52280135588627d017a2191ce2e6605742
google-auth==1.23.0
google-auth-oauthlib==0.4.2
grpcio==1.33.2
idna==2.10
imagesize==1.2.0
importlib-metadata==2.0.0
kiwisolver==1.3.1
markdown==3.3.3
matplotlib==3.3.2
mkl-fft==1.2.0
mkl-random==1.1.1
mkl-service==2.3.0
numpy @ file:///C:/ci/numpy_and_numpy_base_1603468620949/work
oauthlib==3.1.0
olefile==0.46
Pillow @ file:///C:/ci/pillow_1603821929285/work
portalocker==2.0.0
protobuf==4.0.0rc2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools @ git+https://github.com/philferriere/cocoapi.git@2929bd2ef6b451054755dfd7ceb09278f935f7ad#subdirectory=PythonAPI
pycparser @ file:///tmp/build/80754af9/pycparser_1594388511720/work
pyparsing==3.0.0b1
python-dateutil==2.8.1
pywin32==228
PyYAML==5.3.1
requests==2.24.0
requests-oauthlib==1.3.0
rsa==4.6
six==1.15.0
tabulate==0.8.7
tensorboard==2.3.0
tensorboard-plugin-wit==1.7.0
termcolor==1.1.0
torch==1.3.1
torchvision==0.4.2
tqdm==4.51.0
urllib3==1.25.11
werkzeug==1.0.1
wincertstore==0.2
yacs==0.1.8
zipp==3.4.0
```



