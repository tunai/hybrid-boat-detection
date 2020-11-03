"""
File name: main_e2e_only.py
Author: Tunai P. Marques
Website: tunaimarques.com | github.com/tunai
Date created: Jul 15 2020
Date last modified: Oct 14 2020

DESCRIPTION: reads all ".jpg" and ".png" files from sub-folders of the main folder (config.root) and
performs the a marine vessel detection using only end-to-end object detectors.

If this software proves to be useful to your work, please cite: "Tunai Porto Marques, Alexandra Branzan Albu,
Patrick O'Hara, Norma Serra, Ben Morrow, Lauren McWhinnie, Rosaline Canessa. Robust Detection of Marine Vessels
from Visual Time Series. In The IEEE Winter Conference on Applications of Computer Vision, 2021."

"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from utils.utils import *
from utils.utils_XLSX import *
from utils.utils_plotting import *
from config import init_config
import time
import os, shutil, sys

# Reads an performs detection on multiple folders at a time. The results are saved in
# another folder: "positive" for images w/ detected boats, "negative" for images without.
# the folders' names must be numbers.

# MODEL ZOO: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

if __name__ == '__main__':

    print('Version of Detectron2 being used: {}'.format(detectron2.__version__))
    print('Processing multiple folders using end-to-end deep learning-based object detectors only.')

    config = init_config()  # load all parameters from the config.py file

    # grab class names from COCO and set detection threshold
    classNames = MetadataCatalog.get("coco_2014_train").thing_classes
    validRange = [config.upper_ylimit,config.OD_ylimit] # any detection with y-coord outside this range is ignored

    cfg = get_cfg()

    merge_cfg, modelWeights, ODmodelName = pickODModelParameters(config.OD_model_number)

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(merge_cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.OD_detection_threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = modelWeights

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # create a workbook and a worksheet to save the detection results
    site = config.site_name

    # initiate some detection-related parameters
    names = []
    resizeSaturna = 0.5
    outImgSaveType = 1 # 0 = no output images. 1 = image w/ filtered detections. 2 = image w/ all detections + filtered

    # select the input and output directories (note that the spreadsheet always goes to the root folder)

    root = config.root

    names = [x[0] for x in os.walk(root)]
    subfolderNames = names[1:]

    for foldInd in range(0,len(subfolderNames)):

        inDirectory = subfolderNames[foldInd]+'/'
        outDirectory = inDirectory + "processed/"

        print('{}. PROCESSING FOLDER {} -> OUT ON {}'.format(foldInd+1,inDirectory,outDirectory))

        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
            print("Created an output folder.")
        else:
            sys.exit('There exists a "processed" subfolder in one of the folders. '
                     'Please remove it to prevent data from being overwritten.')

        workbook, worksheet = createXLSX(site)
        rowCounter = 1

        # read all files from a folder and start detection
        allfiles = sorted(os.listdir(inDirectory))
        files = []
        for f in allfiles:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPG"):
                files.append(f)

        numFiles = len(files)

        # for each file,
        for i in range(0, numFiles):

            # get the current file and read the image
            current = files[i]
            print('Processing file {} ({}) from {}'.format(i, current, numFiles))
            address = inDirectory + current
            image = cv2.imread(address)
            image2 = cv2.imread(address)

            # resize the image if it comes from Saturna island
            if (site == 'Saturna'):
                image = cv2.resize(image, (int(image.shape[1] * resizeSaturna), int(image.shape[0] * resizeSaturna)),
                                   interpolation=cv2.INTER_AREA)

            validRange[1] = image.shape[0]

            # perform the detection and calculate the processing time
            start_time = time.time()
            outputs = predictor(image)
            print("detection time = {} seconds".format(time.time() - start_time))

            # filter and concatenate the detection results and prepare the strings to be saved
            filt_bboxes, filt_scores, filt_class, detString, scoreString = postProcessDetections(outputs,config.OD_detection_threshold, validRange)
            # plotAllBB(image2, filt_bboxes, score=filt_scores, classes=filt_class)

            numBoats = filt_bboxes.__len__()

            # save an annotated version of the image (image+detections)
            if outImgSaveType == 1:
                readAndPlotOD(filt_bboxes, filt_scores, filt_class, image, current, outDirectory)
            elif outImgSaveType == 2:
                readAndPlotOD(filt_bboxes, filt_scores, filt_class, image, current, outDirectory)
                readAndPlotODAll(outputs, classNames, image, current, outDirectory)
            else:
                pass

            # update the worksheet
            worksheet = updateXLSX(worksheet, rowCounter, inDirectory, current, site, numBoats, det=detString, scores=scoreString)
            rowCounter += 1

        # close the workbook once the detections are over
        workbook.close()
        shutil.move(workbook.filename, outDirectory)
