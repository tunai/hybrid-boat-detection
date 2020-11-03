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

### Installation

1. **Detectron2**. The detector was built using [conansherry's Windows build of Detectron2](https://github.com/conansherry/detectron2). Follow the repo's intructions to install it. Although not tested, it should work with the original Linux version. Once you are done, run on your Python environment: 
        
```python
import detectron2
detectron2.__version__
```
You should be able to see the version of your Detectron2. 

2.
```git clone https://github.com/tunai/hybrid-boat-detection
cd hybrid-boat-detection
conda install pip
```
        



### Repo author

Tunai Porto Marques (tunaip@uvic.ca), [website](https://www.tunaimarques.com) 



