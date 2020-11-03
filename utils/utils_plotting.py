"""
File name: utils_plotting.py
Author: Tunai P. Marques
Website: tunaimarques.com | github.com/tunai
Date created: Jul 01 2020
Date last modified: Nov 02 2020

DESCRIPTION: implements a number of plotting functions to support the hybrid marine vessels detector. except for
research purposes, we recommend that users do not modify this script.

If this software proves to be useful to your work, please cite: "Tunai Porto Marques, Alexandra Branzan Albu,
Patrick O'Hara, Norma Serra, Ben Morrow, Lauren McWhinnie, Rosaline Canessa. Robust Detection of Marine Vessels
from Visual Time Series. In The IEEE Winter Conference on Applications of Computer Vision, 2021."

"""


import cv2
from datetime import datetime
import numpy as np
import os

def plotAllBB(img,bb,color=(0,255,255),line=1,display=None, score=None, classes=None, title="All Bounding Boxes"):

    placeHolderImg = img.copy()

    for i in range(0,len(bb)):

        if bb[0].shape.__len__() == 2:  # lists of [[np.array]]
            current = bb[i][0]
        else:
            current = bb[i]

        print('{}:{},{},{},{}'.format(i, current[0], current[1], current[2], current[3]))
        cv2.rectangle(placeHolderImg, (current[0], current[1]), (current[2], current[3]), color, line)

        if (score is not None) and (classes is not None):
            cv2.putText(placeHolderImg, (str(round(score[i], 2)) + " C " + str(classes[i])), (current[0] - 10, current[1] - 10), 1,
            1, (0, 255, 255), 1)

    if display is not None:
        cv2.namedWindow(title, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(title, 0, 0)
        cv2.imshow(title, placeHolderImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return placeHolderImg

def plotBB_BS_Result(img,bb,color=(0,255,255),line=1,
                     display=None, score=None, classes=None,
             title="All Bounding Boxes", img_id = None):

    placeHolderImg = img.copy()

    for i in range(0,len(bb)):
        current = bb[i]

        cv2.rectangle(placeHolderImg, (current[0], current[1]), (current[2], current[3]), color, line)

        if (score is not None) and (classes is not None):
            cv2.putText(placeHolderImg, (str(round(score[i], 2)) + " C " + str(classes[i])), (current[0] - 10, current[1] - 10), 1,
            1, (0, 255, 255), 1)

        if img_id is not None:
            cv2.putText(placeHolderImg, img_id, (current[0] - 10, current[1] - 10), 1, 1, (0, 255, 255), 1)

    if display is not None:
        cv2.namedWindow(title, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(title, 0, 0)
        cv2.imshow(title, placeHolderImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return placeHolderImg

def showIMG(img, title = "image display"):

    if img.shape.__len__() == 2:
        if img.max() <= 1:
            print("multiplying by 255 first...")
            img = img * 255
        img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)

    if img.dtype == 'int32':
        img = img.astype(np.uint8)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readAndPlotOD(det, score, classes, image, names, outPath, outNameP = "/positive/", outNameN = "/negative/"):
    # function to read the output of OD detection, plot them on an image and save the image
    # note: specific to detectron2 output format

    if not os.path.exists(outPath+outNameP):
        os.makedirs(outPath+outNameP)

    if not os.path.exists(outPath+outNameN):
        os.makedirs(outPath+outNameN)

    for i in range(0,len(det)):

        prob = score[i]
        currentClass = classes[i]

        if det[0].shape.__len__() == 2:  # lists of [[np.array]]
            current = det[i][0]
        else:
            current = det[i]

        colorOD = (0, 255, 255)
        colorSMVD = (0, 0, 255)

        if prob == 0.1: # change the bb color if detection comes from SMVD
            cv2.rectangle(image, (current[0], current[1]), (current[2], current[3]), colorSMVD, 1)
        else:
            cv2.rectangle(image, (current[0], current[1]), (current[2], current[3]), colorOD, 1)

        cv2.putText(image, (str(round(prob, 4)) + "C " + str(currentClass)), (current[0] - 10, current[1] - 10), 1, 2, (0, 255, 255), 1)

    scale = 1  # rescale if necessary
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    nameBefore = names.split('.jpg')[0]
    suffix = "_ODOUT.jpg"

    if len(det) > 0:
        cv2.imwrite(outPath + outNameP + nameBefore + suffix, image)
    else:
        cv2.imwrite(outPath + outNameN + nameBefore + suffix, image)

def readAndPlotMerged(det, score, classes, image, names, outPath, outNameP = "/positive/", outNameN = "/negative/"):
    # function to read the output of detections, plot them on an image and save the image
    # note: specific to detectron2 output format

    if not os.path.exists(outPath+outNameP):
        os.makedirs(outPath+outNameP)

    if not os.path.exists(outPath+outNameN):
        os.makedirs(outPath+outNameN)

    for i in range(0,len(det)):

        prob = score[i]
        currentClass = classes[i]

        if det[0].shape.__len__() == 2:  # lists of [[np.array]]
            current = det[i][0]
        else:
            current = det[i]

        colorOD = (0, 255, 255)
        colorSMVD = (0, 0, 255)

        if prob == 0.91111: # change the bb color if detection comes from SMVD
            cv2.rectangle(image, (current[0], current[1]), (current[2], current[3]), colorSMVD, 2)
        else:
            cv2.rectangle(image, (current[0], current[1]), (current[2], current[3]), colorOD, 2)

        cv2.putText(image, (str(round(prob, 2))), (current[0] - 10, current[1] - 10), 1, 2, (0, 255, 255), 1)

    scale = 1  # rescale if necessary
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    nameBefore = names.split('.jpg')[0]
    suffix = "_OUT.jpg"

    if len(det) > 0:
        cv2.imwrite(outPath + outNameP + nameBefore + suffix, image)
    else:
        cv2.imwrite(outPath + outNameN + nameBefore + suffix, image)

def readAndPlotODAll(det, classNames, image, names, outPath):

    filt_bboxes = []
    filt_scores = []
    filt_class = []
    targetClass = (8, 4)

    classes = det['instances']._fields['pred_classes']
    scores = det['instances']._fields['scores']
    bboxes = det['instances']._fields['pred_boxes']

    # first turn the results into numpy arrays and add them to list
    for i in range(0,len(classes)):

        # get the tensor, transfer it to cpu, turn to numpy array and typecast to int
        bbox = bboxes[i].tensor.cpu().numpy().astype(int)
        classCurrent = classes[i].cpu().numpy().astype(int)
        # get the tensor, transfer it to cpu, turn to numpy array and round it to 2 decimal places
        score = scores[i].cpu().numpy().round(2)
        filt_bboxes.append(bbox)
        filt_scores.append(score)
        filt_class.append(classCurrent)

    for i in range (0, len(filt_bboxes)):
        prob = round(filt_scores[i],2)

        currentClass = filt_class[i]
        if currentClass in targetClass:
            color = (0,0,255)
        else:
            color = (0, 255, 255)

        nameClass = classNames[currentClass]
        string = str(prob) + " " + str(nameClass) + "(" + str(currentClass) + ")"
        current = filt_bboxes[i][0]
        cv2.rectangle(image, (current[0], current[1]), (current[2], current[3]), color, 1)
        cv2.putText(image, string, (current[0] - 10, current[1] - 10), 1, 2, (0, 255, 255), 2)

    scale = 1
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    nameBefore = names.split('.jpg')[0]
    # print(nameBefore)
    suffix = "_ODOUT_ALL.jpg"
    cv2.imwrite(outPath + nameBefore + suffix, image)

def plotClassificationResults(imgs, preds, bbs, name):

    blend = cv2.addWeighted(imgs[0], 0.5, imgs[1], 0.5, 0.0)
    blend = cv2.addWeighted(blend, 0.66, imgs[2], 0.33, 0.0)

    for i in range(len(bbs)):
        current = bbs[i]
        bwd = current[:,0]
        mid = current[:,1]
        fwd = current[:,2]

        if preds[3*i] == 1:
            cv2.rectangle(blend, (bwd[0], bwd[1]), (bwd[0] + bwd[2], bwd[1] + bwd[3]), (3, 3, 252), 1)
            cv2.putText(blend, 'V', (bwd[0] - 5, bwd[1] - 5), 1, 1, (3, 3, 252), 1)
        else:
            cv2.rectangle(blend, (bwd[0], bwd[1]), (bwd[0] + bwd[2], bwd[1] + bwd[3]), (0, 255, 255), 1)
            cv2.putText(blend, 'B', (bwd[0] - 5, bwd[1] - 5), 1, 1, (0, 255, 255), 1)

        if preds[1 + 3*i] == 1:
            cv2.rectangle(blend, (mid[0], mid[1]), (mid[0] + mid[2], mid[1] + mid[3]), (3, 3, 252), 1)
            cv2.putText(blend, 'V', (mid[0] - 5, mid[1] - 5), 1, 1, (3, 3, 252), 1)
        else:
            cv2.rectangle(blend, (mid[0], mid[1]), (mid[0] + mid[2], mid[1] + mid[3]), (0, 255, 255), 1)
            cv2.putText(blend, 'B', (mid[0] - 5, mid[1] - 5), 1, 1, (0, 255, 255), 1)

        if preds[2 + 3*i] == 1:
            cv2.rectangle(blend, (fwd[0], fwd[1]), (fwd[0] + fwd[2], fwd[1] + fwd[3]), (3, 3, 252), 1)
            cv2.putText(blend, 'V', (fwd[0] - 5, fwd[1] - 5), 1, 1, (3, 3, 252), 1)
        else:
            cv2.rectangle(blend, (fwd[0], fwd[1]), (fwd[0] + fwd[2], fwd[1] + fwd[3]), (0, 255, 255), 1)
            cv2.putText(blend, 'B', (fwd[0] - 5, fwd[1] - 5), 1, 1, (0, 255, 255), 1)

    cv2.imwrite(name, blend)

def blendImages(img1,img2,img3=None):

    if img3 is None:
        blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
        showIMG(blend)
    else:
        blend = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
        blend = cv2.addWeighted(blend, 0.66, img3, 0.33, 0.0)
        showIMG(blend)

def createLog(site, ylimit, modelName, detectionThreshold, mean, std, SMVdetModel, defaultScoreGMM, validRangeValue,
              upperBBlimitValue, thresholdPositives, outDir):

    now = datetime.now()
    now = now.strftime("%d-%m-%Y %H_%M_%S")
    fileName = outDir + now + "_" + "logDetection.txt"
    f = open(fileName, "w+")
    f.write("Experiment time: %s \n" % (now))
    f.write("Site: %s \n\n" % (site))

    f.write("Y-axis Object Detection minimum limit: %d \n" % (ylimit))
    f.write("End-to-end Object Detection model name: %s \n" % (modelName))
    f.write("End-to-end Object Detection threshold: %.2f \n\n" % (detectionThreshold))

    f.write("biGMM classification phase model: %s \n" % (SMVdetModel))
    f.write("Range where the biGMM SMV detection will happen: %s \n" % (str(validRangeValue)))
    f.write("Mean and Std. Deviation used in the transform (biGMM classification phase): %s,%s\n" % (str(mean), str(std)))
    f.write("Default classification score biGMM: %.2f \n" % (defaultScoreGMM))
    f.write("Minimum # of vessels in each group of 3 for it to be valid: %d \n" % (thresholdPositives))
    f.write("Maximum # of motion-triggered BBs per group of 3: %d \n" % (upperBBlimitValue))

    f.close()

    print("")

def throwModelError(model_name, model_path, address):
    print('The specified img. class. model ({}) was not found on "{}".'.format(model_name, model_path))
    print('Download the custom-trained img. class. models 5 and 6 at {}'.format(address))
    print('And place them in the "./configs/Custom-trained Vessel Image Classifiers/" folder.')
    print('Change the "DSMV_img_class_model_number" parameter in the config file to choose another img. class. model')