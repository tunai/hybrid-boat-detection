<img height="171px" width="340px" align="right" src="https://i.imgur.com/r7IpzX8.jpg">  

### Robust Detection of Marine Vessels From Visual Time Series

This repo contains the implementation of our work: **Robust Detection of Marine Vessels From Visual Time Series**, by Tunai Porto Marques *et al.*, accepted for presentation at the 2021 Winter Conference on Applications of Computer Vision (WACV, January 5-9, 2021). The camera-ready version of the manuscript will be released soon. 

If the software provided proves to be useful to your work, please cite its related publication: 

#### BibTeX

>    @inProceedings{TPMarques_WACV_2021,    
>      title={Robust Detection of Marine Vessels from Visual Time Series},    
>      author={Porto Marques, Tunai and Branzan Albu, Alexandra and O'Hara, Patrick and Serra, Norma and Morrow, Ben and McWhinnie, Lauren and Canessa, Rosaline},    
>      booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision},      
>      year={2021}}

### Installation and requirements

1. **Detectron2**. The detector was built using Facebook's Dectron2. More specifically, [conansherry's Windows build of Detectron2](https://github.com/conansherry/detectron2). Follow that repo's intructions on how to install Detectron2, [PyTorch](https://pytorch.org/get-started/locally/), torchvision, [OpenCV](https://anaconda.org/conda-forge/opencv), [fvcore](https://github.com/facebookresearch/fvcore), [pycocotools](https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI) and cython. I have also put together a [tutorial on how to install Detectron2 on Windows](https://github.com/tunai/hybrid-boat-detection/blob/master/install_detectron2_win10.md) with specific package versions which you can follow (both guides will work).  
Although not yet tested, the boat detector should work with the original Linux release of Detectron2. Once you are done, check the installation on your Python environment: 
        
```python
import detectron2
detectron2.__version__
```
You should be able to see the version of your Detectron2 (e.g., "0.1") if the installation was sucessful. 

2. Clone and install this repo and its supporting packages (with your Detectron2 python environment activated on anaconda prompt):
```
git clone https://github.com/tunai/hybrid-boat-detection
cd hybrid-boat-detection
conda install -c anaconda xlsxwriter
```
3. Test the hybrid boat detector:
```python
python main.py
```       
Your output should look like [this](https://i.imgur.com/IadQOxX.jpg) 

### Repo author

Tunai Porto Marques (tunaip@uvic.ca), [website](https://www.tunaimarques.com) 



