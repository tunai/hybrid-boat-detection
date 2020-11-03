"""
File name: main_github.py
Author: Tunai P. Marques
Website: tunaimarques.com | github.com/tunai
Date created: Jul 15 2020
Date last modified: Nov 02 2020

DESCRIPTION: main script. reads all ".jpg" and ".png" files from sub-folders of the main folder (config.root) and
performs the hybrid marine vessel detection using the novel strategy proposed.

INSTRUCTIONS:
0) Change the values on the "config.py" file to reflect your configuration preferences.
1) The file names must follow the "prefix+YYYY-MM-DD_HH-MM-SS.format" template, and groups of three images
 temporally close to each other (we used 5 seconds, but you can try other time intervals) must be provided.
2) The folder(s) being processed must only contains image files and possibly "output sub folders" created by
this script. Any other sub folders will create unpredictable output.

If this software proves to be useful to your work, please cite: "Tunai Porto Marques, Alexandra Branzan Albu,
Patrick O'Hara, Norma Serra, Ben Morrow, Lauren McWhinnie, Rosaline Canessa. Robust Detection of Marine Vessels
from Visual Time Series. In The IEEE Winter Conference on Applications of Computer Vision, 2021."

"""

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision import transforms
from utils.utils import postProcessDetections, pickODModelParameters, pickDSMVModelParameters, createStrings,\
    filterDSMVresults, concatenateBB_OD_DSMV, mergeODandDSMV, createGroupsofThree
from utils.utils_XLSX import createXLSX, updateXLSX
from utils.utils_plotting import readAndPlotOD, readAndPlotMerged, showIMG, throwModelError
from utils.bigmm2 import DSMV
from config import init_config
from datetime import datetime

import numpy as np
import shutil, sys, copy, os, time
import cv2
import torch

if __name__ == '__main__':

    # # uncomment to visualise a sample image and determine the detection bands (i.e., x- and y-limits for OD and DSMV)
    # # change the upper_ylimit, DSMV_ylimit, OD_ylimit and xlimit parameters on the config to adjust to new boundaries.
    # img = cv2.imread('./data/3/capture_2020-07-30_10-01-20(RCNX8249.JPG).jpg')
    # showIMG(img)

    config = init_config()  # load all parameters from the config.py file

    start_time_global_global = time.time()
    print('Processing multiple folders with the Hybrid Marine Vessel detection system.')

    names = [x[0] for x in os.walk(config.root)]
    subfolderNames = names[1:]
    subfolderNames = [x for x in subfolderNames if not "output" in x]  # ignores all output sub folders from folders

    validRangeGMMValue = [config.upper_ylimit,
                          config.DSMV_ylimit]  # range inside which the DSMV will perform detection.
    # usually focused on the top part of the image (smaller boats)
    validRangeODValue = [config.upper_ylimit,
                         config.OD_ylimit]  # range inside which the end-to-end object detector will perform detection.
    # usually focused on the lower part of the image (mid- and large-sized boats)

    # initialize the end-to-end object detector selected in the config
    cfg = get_cfg()  # initializes a standard detectron2 config shell file
    merge_cfg, ODmodelWeights, ODmodelName = pickODModelParameters(config.OD_model_number)
    cfg.merge_from_file(merge_cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.OD_detection_threshold
    cfg.MODEL.WEIGHTS = ODmodelWeights
    predictor = DefaultPredictor(cfg)

    # mean and std values come from the defaults used by models from torchvision (final stage of the DSMV):
    # https://pytorch.org/docs/stable/torchvision/models.html
    # the custom-trained vessel image classifier uses transfer learning, reason why the normalization has
    # to agree with the values originally used to train the rest of these CNN architectures
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # define a transform that turn patches into tensors, and normalizes them with the set mean and std dev
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    DSMVmodel_path, DSMVmodel_name = pickDSMVModelParameters(config.DSMV_img_class_model_number)

    if not (os.path.exists(DSMVmodel_path)):
        throwModelError(DSMVmodel_name, DSMVmodel_path, config.large_models_web_address)
        sys.exit()

    model = torch.load(DSMVmodel_path)
    SMVdetModel = str(model.__class__.__name__)
    model.eval()
    print('-' * 80)
    print('Initialized the DSMV with a custom-trained {} image classifier model and '
          'the end-to-end OD with a pre-trained {} model'.format(DSMVmodel_name, ODmodelName))
    print('-' * 80)

    for foldInd in range(0, len(subfolderNames)):
        inDirectory = subfolderNames[foldInd] + '/'
        now = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
        outDirectory = inDirectory + "/output " + now + "/DSMV/"
        os.makedirs(outDirectory)

        print('\n{}. Processing folder "{}".'.format(foldInd + 1, inDirectory))

        allfiles = sorted(os.listdir(inDirectory))
        files = []
        # grab only the files of a specific format (jpg or png)
        for f in allfiles:
            if f.endswith(".jpg") or f.endswith(".png"):
                files.append(f)

        # the DSMV works on groups of three images at a time. each minute offers 3 images,
        # and they must be grouped following the naming convention:"prefix+YYYY-MM-DD_HH-MM-SS"
        groups_of_three = createGroupsofThree(files, inDirectory, config.prefix)
        print('Number of groups of three images identified in this folder: {}'.format(groups_of_three.__len__()))

        # create an .XLSX spreadsheet to save the results of both hybrid (combined) and OD results
        workbook, worksheet = createXLSX(config.site_name, name="HYBRID")
        workbookOD, worksheetOD = createXLSX(config.site_name, name="OD")
        rowCounter = 1
        count = 1

        start_time_global = time.time()

        for i in groups_of_three:

            filt_bboxes, filt_scores, filt_class = ([] for w in range(3))  # initialize empty lists

            print('Group {} out of {} (MinuteID: {})'.format(count, len(groups_of_three), i["uniqueIDMinute"]))
            count += 1

            if len(i) < 4:  # keep only groups with three images
                groups_of_three.remove(i)
            else:  # for each group of three images, detect using the object detector and the DSMV
                images = [cv2.imread(i["image1"]), cv2.imread(i["image2"]), cv2.imread(i["image3"])]

                # detection using the end-to-end object detector previously chosen
                outputBWD, outputMID, outputFWD = (predictor(images[im]) for im in range(3))

                # filter and concatenate the obj. detection detection results and prepare the strings to be saved
                for det in [outputBWD, outputMID, outputFWD]:
                    filtBboxes, filtScores, filtClass, \
                    detString, scoreString = postProcessDetections(det,config.OD_detection_threshold,validRangeODValue)

                    # print("OD scores: {}".format(filtBboxes))
                    filt_bboxes.append(filtBboxes)
                    filt_class.append(filtClass)
                    filt_scores.append(filtScores)

                for phase in range(3):  # phases are BWD, MID, FWD
                    id = "image" + str(phase + 1)  # phase ID
                    name = i[id][len(i[id]) - (len(config.prefix) + 23):]  # 23 = exact len of "YYYY-MM-DD_HH-MM-SS.jpg"
                    readAndPlotOD(filt_bboxes[phase], filt_scores[phase], filt_class[phase], images[phase].copy(), name,
                                  inDirectory, outNameN="/output " + now + "/negativeOD/", outNameP="/output " + now + "/positiveOD/")

                # executes the DSMV to detect small marine vessels in groups of three temporally close images
                result, blend = DSMV(images[0], images[1], images[2],
                                     outputBlendImg=config.DSMV_outputBlendImg,
                                     validRange=validRangeGMMValue,
                                     upperBBlimit=config.DSMV_upperBBlimit,
                                     debugMode=config.DSMV_debug,
                                     pixelThresh=config.DSMV_pixel_thresh,
                                     xLimit=config.xlimit)

                # deep copy OD results so that they can be modified without changing the original lists
                filt_bboxes_OD = copy.deepcopy(filt_bboxes)
                filt_scores_OD = copy.deepcopy(filt_scores)

                if result is not 0:  # if the DSMV identified any marine vessels,

                    # filter out results not classified as vessels, or for number of vessels lower than the threshold
                    result = filterDSMVresults(i, result, blend, outDirectory, images, transform, model,
                                               config.DSMV_positives_threshold)

                    if result is not 0:  # if there are DSMV results remaining after the filtering, merge them with the
                        # object detection results

                        mergedBBs, mergedScores = mergeODandDSMV(filt_bboxes, filt_scores, result)
                        for idx in range(0, 3):

                            # concatenate the DSMV- and OD-generated detection bounding boxes (BB). The OD detection are
                            # prioritized, thus in case of overlap, the BBs are merged and the score of the OD is kept.
                            mergedBBs[idx], mergedScores[idx], concatClasses = \
                                concatenateBB_OD_DSMV(mergedBBs[idx], mergedScores[idx],
                                                      defaultGMMScore=config.default_DSMV_score)
                else:
                    # if DSMV result is empty, just copy the OD results to the output variables
                    mergedBBs = filt_bboxes
                    mergedScores = filt_scores

                for phase in range(3):  # for each of the three files in a group,

                    # create a "classes" variable to conform with the plot function
                    classes = [8] * len(mergedScores[phase])

                    id = "image" + str(phase + 1)
                    name = i[id][len(i[id]) - (len(config.prefix) + 23):]  # 23 = exact len of "YYYY-MM-DD_HH-MM-SS.jpg"

                    numBoats = len(mergedBBs[phase])
                    numBoatsOD = len(filt_bboxes_OD[phase])

                    # create hybrid and OD output strings to be added in the .XLSX
                    detString, scoreString = createStrings(mergedBBs[phase], mergedScores[phase])
                    detStringOD, scoreStringOD = createStrings(filt_bboxes_OD[phase], filt_scores_OD[phase])

                    # update the worksheet
                    worksheet = updateXLSX(worksheet, rowCounter, inDirectory, name, config.site_name, numBoats,
                                           detString, scoreString)
                    worksheetOD = updateXLSX(worksheetOD, rowCounter, inDirectory, name, config.site_name,
                                             numBoatsOD, detStringOD, scoreStringOD)

                    rowCounter += 1  # keep track of the current row in the .XLSX file

                    # create the resulting image
                    readAndPlotMerged(mergedBBs[phase], mergedScores[phase], classes, images[phase], name,
                                      inDirectory, outNameN="/output " + now + "/negativeHybrid/", outNameP="/output " + now + "/positiveHybrid/")

                    torch.cuda.empty_cache()  # empty CUDA memory so that it does not run out of memory

        total_time = time.time() - start_time_global
        # once all groups of three are done, close, save and move the spreadsheet
        workbook.close()
        workbookOD.close()
        shutil.move(workbook.filename, inDirectory + "/output " + now + "/")
        shutil.move(workbookOD.filename, inDirectory + "/output " + now + "/")

        if (numBoats is not 0) and (total_time > 0):
            print("Folder processing time = {} s. Time"
                  " per image: {} s".format(round(total_time,2),(total_time / len(groups_of_three)) / 3))

        if config.DSMV_savePatches is False:
            print('Cleaning DSMV folder...')
            shutil.rmtree(inDirectory + 'DSMV/')

    print("Global processing time = {} s.".format(round(time.time() - start_time_global_global),2))