"""
File name: config.py
Author: Tunai P. Marques
Website: tunaimarques.com | github.com/tunai
Date created: Aug 10 2020
Date last modified: Nov 02 2020

DESCRIPTION: configuration file to determine the values of parameters from the main execution script,
end-to-end object detector and DSMV.

INSTRUCTIONS: change the parameters of this file to modify the behaviour of the detector. See "help" fields of
each parameter for details.

If this software proves to be useful to your work, please cite: "Tunai Porto Marques, Alexandra Branzan Albu,
Patrick O'Hara, Norma Serra, Ben Morrow, Lauren McWhinnie, Rosaline Canessa. Robust Detection of Marine Vessels
from Visual Time Series. In The IEEE Winter Conference on Applications of Computer Vision, 2021."

"""

import argparse

lists = []
parser = argparse.ArgumentParser()

def new_group(name):
    arg = parser.add_argument_group(name)
    lists.append(arg)
    return arg

def init_config():
    config, unparsed = parser.parse_known_args()
    return config


# Parameters for the execution of the main program
main_arg = new_group("Main")
main_arg.add_argument("--root", type=str,
                      default="./data/",
                      help="Specifies where the folders with the data are. The"
                           " folders in this directory will be visited sequentially.")

main_arg.add_argument("--site_name", type=str,
                      default="Test Site",
                      help="name of the site where the detection take place.")

main_arg.add_argument("--prefix", type=str,
                      default="sample_scene_",
                      help="prefix of all images to be processed. The images' titles must follow this template:"
                           "prefix+YYYY-MM-DD_HH-MM-SS")

main_arg.add_argument("--large_models_web_address", type=str,
                      default="https://drive.google.com/drive/folders/1bO75cOuDVWXC6opNJT37usga1YOe6M_D?usp=sharing",
                      help="web address of larger custom-trained img. class. models. Place them on the "
                           "./config/Custom-trained Vessel Image Classifiers/ folder after download.")

# Parameters from the DSMV and OD detection frameworks
detection_arg = new_group("Detection")

detection_arg.add_argument("--OD_model_number", type=int,
                           default=5,
                           choices=[1, 2, 3, 4, 5],
                           help="Determines the end-to-end object detection architecture to be used"
                                "(pre-trained and offered by Facebook's detectron2):"
                                "1: F-RCNN R-101 FPN 3X, 2: F-RCNN R-50 FPN 3X, 3: F-RCNN X101-FPN 3X,"
                                "4: Cascade R-CNN R-50 FPN 3X, 5: RetinaNet R-101 3X")

detection_arg.add_argument("--DSMV_img_class_model_number", type=int,
                           default=2,
                           choices=[1, 2, 3, 4, 5, 6],
                           help="Determines the custom-trained image classification model (last phase of the DSMV)"
                                "to be used: 1: ResNet-50, 2: Inception V3, 3: DenseNet-201, 4: ResNext-50,"
                                "5: ResNext-101, 6:Wide ResNet-50")

detection_arg.add_argument("--upper_ylimit", type=int,
                           default=192,
                           help="detection with top-left y-coord lower than that is ignored by both the OD and DSMV.")

detection_arg.add_argument("--xlimit", type=int,
                           default=0,
                           help="detection with top-left x-coord lower than that is ignored by both the OD and DSMV.")

detection_arg.add_argument("--DSMV_ylimit", type=int,
                           default=386,
                           help="detection with top-left y-coord higher than that is ignored by the DSMV.")

detection_arg.add_argument("--OD_ylimit", type=int,
                           default=644,
                           help="detection with top-left y-coord higher than that"
                                "is ignored by the object detector.")

detection_arg.add_argument("--default_DSMV_score", type=float,
                           default=0.91111,
                           help="score associated with a DSMV detection. used for the AP calculations.")

detection_arg.add_argument("--OD_detection_threshold", type=float,
                           default=0.2,
                           help="detection threshold for the end-to-end object detection frameworks.")

detection_arg.add_argument("--DSMV_savePatches", type=bool,
                           default=True,
                           help="flag to determine if DSMV-generated motion-triggered patches"
                                "are saved in the output folder.")

detection_arg.add_argument("--DSMV_outputBlendImg", type=bool,
                           default=True,
                           help="flag to determine if blended images and patches"
                                " from the DSMV are saved in an output folder.")

detection_arg.add_argument("--DSMV_upperBBlimit", type=int,
                           default=18,
                           help="maximum number of groups of three motion-triggered sets of CC"
                                "to be considered during the DSMV detection.")

detection_arg.add_argument("--DSMV_pixel_thresh", type=int,
                           default=45,
                           help="motion-triggered sets of connected components with a number of pixels lower than"
                                "this threshold are ignored by the DMSV.")

detection_arg.add_argument("--DSMV_positives_threshold", type=int,
                           default=3,
                           choices=[1, 2, 3],
                           help="number of bounding boxes classified as boats in a group of FMBB-MIBB-BMBB"
                                "for it to be considered valid")

detection_arg.add_argument("--DSMV_debug", type=bool,
                           default=False,
                           help="flag to determine if the debug steps of the DSMV are executed.")
