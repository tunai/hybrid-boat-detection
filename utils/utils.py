import cv2
import xlsxwriter
from datetime import datetime
import torch
import numpy as np
from utils_plotting import *


def postProcessDetections(det, threshold, validRange=0):

    # 1. Filter the detectron2 detections based on a threshold
    filt_bboxes, filt_scores, filt_class = filterDetection(det, threshold, validRange)

    # 2. Concatenate the results
    concatBBs, concatScores, concatClasses = concatenateBBs(filt_bboxes.copy(), filt_scores.copy(), classes = filt_class.copy())

    # 3. Create the strings
    detString, scoreString = createStrings(concatBBs, concatScores)

    return concatBBs, concatScores, concatClasses, detString, scoreString

def concatenateBBs(input, scores, classes = None, threshold = 0.1):

    # algorithm: the first bbox of the list is compared with each of the subsequent bboxes. if there is an overlap
    # larger than the threshold, the two bbs are combined. then the two original bboxes are excluded form the list, and
    # the new, concatenated one is placed in the first index of the list. when a bbox does not overlap with anyone
    # else, it is added to the "concatenated" list (meaning that there is no more combining to be done with it) and
    # excluded from the original list. all bboxes will eventually get to this condition, moment when the algorithm
    # stops (because the original list is empty).

    if classes is None:
        classes = [np.array(8)]*len(scores)

    nboxes = input.__len__()
    boxes = []
    for i in range(0,nboxes):
        if len(input[0].shape)==1:
            boxes.append(input[i])
        else:
            boxes.append(input[i][0])

    concatenated = []
    concatenatedScores = []
    concatenatedClasses = []

    # print('Will start with {}'.format(boxes))

    # concatenate all the entries that overlap in "boxes"
    while boxes:
        concat = 0  # determines if boxes[0] overlaps with any other BB
        if boxes.__len__() > 1:  # if there are at least two non-processed bboxes
            for i in range (1,boxes.__len__()):
                # print('-----------Started to evaluate one BB w/ boxes {} i = {}'.format(boxes,i))
                boxA = boxes[0]
                boxB = boxes[i]
                # print('{}/{}'.format(boxA,boxB))
                iou = ioUtwoBBs(boxA, boxB)
                # print('IoU between boxes {}:{} and {}:{} = {}'.format(0, boxA, i, boxB, iou))
                if iou > threshold:
                    # print("append these two!")

                    # removes the two overlapping BBs from "boxes"
                    # print("boxes before exclusion: {}".format(boxes))
                    boxes.pop(i)
                    boxes.pop(0)

                    # update the score and class of position 0 to reflect the concatenated result
                    if scores[i] > scores[0]:
                        scores[0] = scores[i]
                        classes[0] = classes[i]
                    else:
                        pass  # in this case, the "classes[0]" and "score[0]" do not need to change

                    # change the bboxes by the new, concatenated one
                    newBBox = np.array([(min(boxA[0],boxB[0])), (min(boxA[1],boxB[1])), (max(boxA[2],boxB[2])), (max(boxA[3],boxB[3]))])
                    # print("new concatenated bbox: {}".format(newBBox))
                    boxes.insert(0,newBBox)

                    # print("boxes after exclusion and insertion: {}".format(boxes))
                    concat = 1
                    break

            # if bb index 0 did not have any overlap with anyone, take it out of "boxes" and move it to the final list
            if concat == 0:
                # print('Adding {} and score {} to result'.format(boxes[0],scores[0]))
                concatenated.append(boxes[0])
                concatenatedScores.append(scores[0])
                concatenatedClasses.append(classes[0])
                boxes.pop(0)
                scores.pop(0)
                classes.pop(0)
                # print('boxes after excluding one for concat. {}'.format(boxes))

        # when there is only one bbox left, add it to the final list and take it out of "boxes" (so that the loop is
        # over)
        else: # the elements must but taken out of "boxes", otherwise the "while" does not end
            # print('Adding {} and score {} to result'.format(boxes[0],scores[0]))
            concatenated.append(boxes[0])
            concatenatedScores.append(scores[0])
            concatenatedClasses.append(classes[0])
            boxes.pop(0)
            scores.pop(0)
            classes.pop(0)

    return concatenated, concatenatedScores, concatenatedClasses

def concatenateBB_OD_DSMV(input, scores, classes = None,
                        threshold = 0, defaultGMMScore = 0.9111):

    # algorithm: the first bbox of the list is compared with each of the subsequent bboxes. if there is an overlap
    # larger than the threshold, the two bbs are combined. then the two original bboxes are excluded form the list, and
    # the new, concatenated one is placed in the first index of the list. when a bbox does not overlap with anyone
    # else, it is added to the "concatenated" list (meaning that there is no more combining to be done with it) and
    # excluded from the original list. all bboxes will eventually get to this condition, moment when the algorithm
    # stops (because the original list is empty).

    # numODbb = len(finalODresult)
    # numGMMbb = finalGMMresult.shape[1]

    # source = [1]*numODbb
    # source.extend([0]*numGMMbb)

    if classes is None:
        classes = [np.array(8)]*len(scores)

    nboxes = input.__len__()
    boxes = []
    for i in range(0,nboxes):
        if len(input[0].shape)==1:
            boxes.append(input[i])
        else:
            boxes.append(input[i][0])

    # print("initial bbs:{}".format(boxes))

    concatenated = []
    concatenatedScores = []
    concatenatedClasses = []

    # print('Will start with {}'.format(boxes))

    # concatenate all the entries that overlap in "boxes"
    while boxes:
        concat = 0  # determines if boxes[0] overlaps with any other BB
        if boxes.__len__() > 1:  # if there are at least two non-processed bboxes
            for i in range (1,boxes.__len__()):
                # print('-----------Started to evaluate one BB w/ boxes {} i = {}'.format(boxes,i))
                boxA = boxes[0]
                boxB = boxes[i]
                # print('{}/{}'.format(boxA,boxB))
                iou = ioUtwoBBs(boxA, boxB)
                # print('IoU between boxes {}:{} and {}:{} = {}'.format(0, boxA, i, boxB, iou))
                if iou > threshold:

                    # print("delete one of them!")

                    # removes the two overlapping BBs from "boxes"
                    # print("boxes before exclusion: {}".format(boxes))
                    boxes.pop(i)
                    boxes.pop(0)
                    # print("boxes after exclusion: {}".format(boxes))

                    # update the score and class of position 0 to reflect the concatenated result
                    if scores[i] > scores[0]:
                        scores[0] = scores[i]
                        classes[0] = classes[i]
                    else:
                        pass  # in this case, the "classes[0]" and "score[0]" do not need to change

                    # always get boxA because the want to emphasize the OD result
                    newBBox = np.array([boxA[0], boxA[1], boxA[2], boxA[3]])

                    # print("new concatenated bbox: {}".format(newBBox))
                    boxes.insert(0,newBBox)

                    # print("boxes after exclusion and insertion: {}".format(boxes))
                    concat = 1
                    break

            # if bb index 0 did not have any overlap with anyone, take it out of "boxes" and move it to the final list
            if concat == 0:
                # print('Adding {} and score {} to result'.format(boxes[0],scores[0]))
                concatenated.append(boxes[0])
                concatenatedScores.append(scores[0])
                concatenatedClasses.append(classes[0])
                boxes.pop(0)
                scores.pop(0)
                classes.pop(0)
                # print('boxes after excluding one for concat. {}'.format(boxes))

        # when there is only one bbox left, add it to the final list and take it out of "boxes" (so that the loop is
        # over)
        else: # the elements must but taken out of "boxes", otherwise the "while" does not end
            # print('Adding {} and score {} to result'.format(boxes[0],scores[0]))
            concatenated.append(boxes[0])
            concatenatedScores.append(scores[0])
            concatenatedClasses.append(classes[0])
            boxes.pop(0)
            scores.pop(0)
            classes.pop(0)

    for idx, item in enumerate(concatenatedScores):
        if item == 0:
            concatenatedScores[idx] = defaultGMMScore

    return concatenated, concatenatedScores, concatenatedClasses

def createStrings(bboxes, scores):

    assert (len(bboxes) == len(scores)), "error: number of scores and classes are different."

    # creates strings based on the final bboxes and scores that are eventually placed in the results spreadsheet
    detString = ""
    scoreString = ""

    for i in range(0, len(bboxes)):
        detString += str(bboxes[i])  # + lastChar
        scoreString += str(scores[i]) + " "

    return detString, scoreString

def filterDetection(det, threshold, validRangeOD=None):
    # this function is designed specifically for the way detectron2 organizes its output

    filt_bboxes = []
    filt_scores = []
    filt_class = []
    targetClass = (8, 4)  # 8 = boats, 4 = airplanes

    classes = det['instances']._fields['pred_classes']
    scores = det['instances']._fields['scores']
    bboxes = det['instances']._fields['pred_boxes']

    for i in range(0, len(classes)):

        # print("Detection {}: class {} score {}".format(i,classes[i],scores[i]))
        if classes[i] in targetClass and scores[i] >= threshold:
            # get the tensor, transfer it to cpu, turn to numpy array and typecast to int
            bbox = bboxes[i].tensor.cpu().numpy().astype(int)
            classCurrent = classes[i].cpu().numpy().astype(int)

            if (validRangeOD is not None) and (bbox[0][1] < validRangeOD[0] or bbox[0][1] > validRangeOD[1]):
                # too high on the image. probably a bird, so just ignore it.
                print('Invalid y-coordinates ({}) on OD! Ignore detection.'.format(bbox[0][1]))
                break

            # get the tensor, transfer it to cpu, turn to numpy array and round it to 2 decimal places
            score = scores[i].cpu().numpy().round(2)
            filt_bboxes.append(bbox)
            filt_scores.append(score)
            filt_class.append(classCurrent)
            # print("Valid! {}".format(bbox))

    return filt_bboxes, filt_scores, filt_class

def pickODModelParameters(modelNumber):

    if modelNumber == 1:
        merge_cfg = "./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        modelWeights = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        modelName = "F-RCNN R-101 FPN 3X"

    elif modelNumber == 2:
        merge_cfg = "./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        modelWeights = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
        modelName = "F-RCNN R-50 FPN 3X"

    elif modelNumber == 3:
        merge_cfg = "./configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        modelWeights = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
        modelName = "F-RCNN X101-FPN 3X"

    elif modelNumber == 4:
        merge_cfg = "./configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        modelWeights = "detectron2://Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl"
        modelName = "Cascade R-CNN R-50 FPN 3X"

    elif modelNumber == 5:
        merge_cfg = "./configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        modelWeights = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl"
        modelName = "RetinaNet R-101 3X"

    return merge_cfg, modelWeights, modelName

def pickDSMVModelParameters(modelNumber):

    # models 3 and 5 can be downloaded at address specified in config.large_models_web_address
    # place them in the './configs/Custom-trained Vessel Image Classifiers/' folder before use.

    if modelNumber == 1:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/resnet_50.pth"
        model_name = "ResNet-50"

    elif modelNumber == 2:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/inception_V3.pth"
        model_name = "Inception V3"

    elif modelNumber == 3:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/densenet_201.pth"
        model_name = "DenseNet-201"

    elif modelNumber == 4:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/resnext_50.pth"
        model_name = "ResNext-50"

    elif modelNumber == 5:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/resnext_101.pth"
        model_name = "ResNext-101"

    elif modelNumber == 6:
        model_path = "./configs/Custom-trained Vessel Image Classifiers/wide_resnet_50.pth"
        model_name = "Wide ResNet-50"

    return model_path, model_name

def CCCalculateandFilter(originalimg, img, pixelThresh=30, display=0, showAreas = False):
    # create binary image
    img[img > 0] = 255

    # calculate the connected components
    out = cv2.connectedComponentsWithStats(img, 8, cv2.CV_16U)

    # sort the results in descending order based on their area (5th column of out[2]). The "[::-1]" inverts the
    # result, making sure that the ascending order turns into descending
    results = out[2][out[2][:, 4].argsort()[::-1]]

    # grab only indexes for CC with # pixels > threshold (and ignore gigantic ones). "results[:, 4]" represents the
    # 5th column of out[2] i.e., the areas
    indexes = [i for i, v in enumerate(results[:, 4]) if (v > pixelThresh and v < 100000)]

    if showAreas and indexes: # only gets in if flag is set and indexes list is not empty
        for idx, i in enumerate(indexes):
            print('CC # {} pixel area: {}'.format(idx, results[i, :][4]))

    # grab the coordinates of the filtered BB (x,y,deltaX,deltaY)
    ccGroup = results[indexes, 0:4]
    if ccGroup.shape[0] == 0:
        print("No valid CC found.")
        ccGroup = 0
        return ccGroup

    if display == 1:
        displayimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in ccGroup:
            print('{},{},{},{}'.format(i[0], i[1], i[2], i[3]))
            cv2.rectangle(displayimg, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 255, 255), 1)
        cv2.imshow('Filtered CC', displayimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ccGroup

def templateMatching(FWDimg, MIDimg, BWDimg, ccFWD, ccBWD, debugMode = False, MSEthresh = 600, xlimit = 0):

    # print('The MSE Threshold is: {}'.format(MSEthresh))

    # create an array to save the results. The last layer (fwd image template) is initially filled
    # result[:,:,0] = fwd results; result[:,:,1] = mid results; result[:,:,0] = bwd results
    result = np.zeros((ccFWD.shape[0], 4, 3), dtype=int)

    # zeroMASK is a binary image with 1's (or 255's) only where there was movement detected before
    zerosMASK = np.zeros(BWDimg.shape, dtype=int)
    code = 255

    # fill zeroMASK with 255's where movement was detected in the backward image. each subsequent connected component
    # will be filled with 255-(n*5), so that we can distinguish between between regions in zerosMASK that represent
    # different sets of movement-calculated connected components
    for i in ccBWD:
        xs, ys, dx, dy = [i[j] for j in (0, 1, 2, 3)]
        zerosMASK[ys:ys + dy, xs:xs + dx] = code
        code -= 5

    # convert zerosMASK to 1 channel for further template matching
    zerosMASK = (np.sum(zerosMASK, axis=2) / 3).astype(np.uint8)

    ########### FOR DEBBUGING PURPOSES (RED -> BACKWARD, YELLOW -> FORWARD)
    # this section highlights the movement-triggered ccs in the bwd and fwd images
    if debugMode:
        zerosFWD = FWDimg.copy()
        zerosBWD = BWDimg.copy()

        for i in ccFWD:
            xs,ys,dx,dy = [i[j] for j in (0,1,2,3)]
            cv2.rectangle(zerosFWD, (xs, ys), (xs + dx, ys + dy), (0,255,255), 1)
        for i in ccBWD:
            xs,ys,dx,dy = [i[j] for j in (0,1,2,3)]
            cv2.rectangle(zerosBWD, (xs, ys), (xs + dx, ys + dy), (0,0,255), 1)

        blendedImg = cv2.addWeighted(zerosBWD, 0.5, zerosFWD, 0.5, 0.0)
        showIMG(blendedImg, title = "DEBUG 1: Filtered CC BWD (red) and FWD (yellow)")

    # # create image mask so that only matches that happen on top of filtered ccs are considered
    # image_mask = image.copy()
    # image_mask[image_mask>0] = 255

    # image = cv2.cvtColor(zeros.astype(np.float32),cv2.COLOR_RGB2GRAY)

    # Apply template Matching

    # choose the method to be used in the template matching
    # method = cv2.TM_CCOEFF
    method = cv2.TM_CCOEFF
    # ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR' (GREAT ONE), 'cv2.TM_CCORR_NORMED' (GREAT ONE),
    # 'cv2.TM_SQDIFF' (GREAT ONE), 'cv2.TM_SQDIFF_NORMED' (GREAT ONE)]

    # xlimit = FWDimg.shape[1]-1
    # the multiplier determines the width of the horizontal band around the target to be considered in the template
    # matching process
    ylimit = FWDimg.shape[0] - 1
    multiplier = 1.4 # horizontal multiplier
    temporalTunnelWmultiplier = 2
    counter = 1

    # create 1-channel version of all images
    fwdIMG1C = (np.sum(FWDimg, axis=2) / 3).astype(np.uint8)
    bwdIMG1C = (np.sum(BWDimg, axis=2) / 3).astype(np.uint8)
    midIMG1C = (np.sum(MIDimg, axis=2) / 3).astype(np.uint8)

    if debugMode:
        print('Number of motion triggered CCs: {}'.format(ccFWD.shape[0]))

    for idx,i in enumerate(ccFWD):  # for each filtered CC of FWD, do template matching with BWD and then MID:

        if debugMode:
            print("-"*30+"\nProcessing FWD CC number {}".format(idx))

        # grab the coordinates of the filtered cc and create the template
        xs, ys, w, h = [i[j] for j in (0, 1, 2, 3)]
        template = fwdIMG1C[ys:ys + h, xs:xs + w]

        # adjust image by only considering a horizontal band around the template whose width is determined by
        # "multiplier"
        newYs = int(max(0, ys - (multiplier * h)))
        newYe = int(min(ylimit, ys + (multiplier * h)))
        imgAdjusted = bwdIMG1C[newYs:newYe,:]

        if i[3] > imgAdjusted.shape[0]:
            pass
        else:

            if debugMode:
                # image_adjusted_display = cv2.rectangle(imgAdjusted.copy(), (0,newYs), (xlimit,newYe), 255, 1)
                showIMG(blendedImg[newYs:newYe,:], title="DEBUG 2: Adjusted image with fwd (yellow) and bwd (red) ccBBs highlighted")

            if debugMode:
                # image_adjusted_display = cv2.rectangle(imgAdjusted.copy(), (0,newYs), (xlimit,newYe), 255, 1)
                showIMG(imgAdjusted,
                        title="DEBUG 3: Adjusted BWD image")

            if debugMode:
                zerosBWDSafe = zerosBWD[newYs:newYe, :]
                showIMG(zerosBWDSafe, title = "DEBUG 4: BWD Image with BBs highlighted")

            # perform template matching in the fwd/bwd pair of images
            res = cv2.matchTemplate(imgAdjusted, template, method)
            res[:,0:xlimit] = 0
            resSafe = res.copy()
            res = (np.interp(res, (res.min(), res.max()), (0, 255))).astype(np.uint8)

            if debugMode:
                showIMG(res, title="DEBUG 5: Original result in the [0,255] range")

            # showIMG(res, title="result")
            # print('1st Template M. BEFORE FILTERING -> MAX: {}/{} MIN: {}/{}'.format(res.max(),resSafe.max(),res.min(),resSafe.min()))

            # adjust the BWD target image to match the dimensions of the template matching result
            # imageCurrent = image[0:image.shape[0] - h + 1, 0:image.shape[1] - w + 1]

            zerosMASKCurrent = zerosMASK[newYs:newYe-h+1, 0:zerosMASK.shape[1]-w+1]

            # zerosMASK[0:zerosMASK.shape[0] - h + 1, 0:zerosMASK.shape[1] - w + 1] = 1
            # showIMG(zerosMASKCurrent, title="MASK TO BE APPLIED HERE")

            # change all the values in the result that are not included in the BWD target image to zero
            res[zerosMASKCurrent == 0] = 255 if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else 0
            resSafe[zerosMASKCurrent == 0] = 255 if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else 0

            # print('1st Template M. AFTER FILTERING -> MAX: {}/{} MIN: {}/{}'.format(res.max(), resSafe.max(), res.min(), resSafe.min()))

            if debugMode:
                showIMG(res, title="DEBUG 6: Result w/ the BWD mask applied")

            # calculate the location of min and max value to find the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # only look for bwd and mid matches if the max value > 0 (i.e., there is a valid bwd match). otherwise, all the
            # detections should be considered invalid and it is not necessary to calculate mid matching
            if max_val > 0:

                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum value in the results
                top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc

                # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                #     top_left = min_loc
                # else:
                #     top_left = max_loc

                # re-adjust the coordinates of top left point coordinates to reflect the original image again
                top_left = (top_left[0], top_left[1]+ newYs)
                bottom_right = (top_left[0] + w, top_left[1] + h)

                # update the results with the coordinates of the bwd image matches
                # exclude from the zerosMASK the regions that were already matched. thus they cannot be match with anymore

                # # # # # # # # Find vessel in the middle image

                # grab the bwd match
                match = bwdIMG1C[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

                # the middle template will be the blend between the original fwd template and its bwd match
                mixed_template = cv2.addWeighted(template, 0.5, match, 0.5, 0.0)

                if debugMode:
                    showIMG(mixed_template, title = "DEBUG 7: Mixed template")

                # calculate the coordinates of the temporal tunnel based on fwd template and bwd match
                tunnel_bottomright = (max(bottom_right[0],(xs+w)),max(bottom_right[1],(ys+h)))
                tunnel_topleft = (min(top_left[0], xs), min(top_left[1], ys))
                tunnel_w, tunnel_h = (abs(tunnel_topleft[0] - tunnel_bottomright[0]),
                                      abs(tunnel_topleft[1] - tunnel_bottomright[1]))

                if debugMode:
                    # visualize all three images and temporal tunnel
                    mixed_all = (cv2.cvtColor(MIDimg,cv2.COLOR_RGB2GRAY)/3)+(cv2.cvtColor(FWDimg,cv2.COLOR_RGB2GRAY)/3)\
                                 +(cv2.cvtColor(BWDimg,cv2.COLOR_RGB2GRAY)/3)
                    cv2.rectangle(mixed_all, tunnel_topleft, (tunnel_topleft[0]+tunnel_w,tunnel_topleft[1]+tunnel_h), 255, 1)
                    cv2.rectangle(mixed_all, top_left, (top_left[0] + w, top_left[1] + h), 255,1)
                    cv2.putText(mixed_all, "B", (top_left[0]-5, top_left[1] - 5), 1, 1, (0, 255, 255),1)
                    cv2.rectangle(mixed_all, (xs,ys), (xs + w, ys + h), 255, 1)
                    cv2.putText(mixed_all, "F", (xs - 5, ys - 5), 1, 1, (0, 255, 255),1)
                    cv2.putText(mixed_all, str(counter), (tunnel_topleft[0] - 15, tunnel_topleft[1] - 5), 1, 1, (0, 255, 255), 1)
                    showIMG(mixed_all, title = "DEBUG 8: three images mixed and temporal tunnel")

                # create the adjusted mid image by only considering the content of the temporal tunnel
                temporalTunnel = midIMG1C[tunnel_topleft[1]:tunnel_topleft[1] + tunnel_h,
                                 tunnel_topleft[0]:tunnel_topleft[0] + tunnel_w]
                if debugMode:
                    showIMG(temporalTunnel, title = "DEBUG 9: temporal tunnel in the mid image")

                # perform template matching in the content of the temporal tunnel using the mixed template
                resMID = cv2.matchTemplate(temporalTunnel, mixed_template, method)
                resMID = (np.interp(resMID, (resMID.min(), resMID.max()), (0, 255))).astype(np.uint8)

                # calculate the mid of the temporal tunnel (plus some allowance)
                allowedW = int(temporalTunnelWmultiplier*w)
                half_allowedW = int(allowedW/2)
                middle_TTunnel = int(tunnel_w/2)
                allowedW_xs = max(0,middle_TTunnel-half_allowedW)
                allowedW_xe = min(tunnel_w, middle_TTunnel + half_allowedW)
                validRangeOnTM = [allowedW_xs,allowedW_xe-w]

                if debugMode:
                    temporalTunnelCopy = temporalTunnel.copy()
                    cv2.rectangle(temporalTunnelCopy,(allowedW_xs,0),(allowedW_xe,tunnel_h),255,1)
                    showIMG(temporalTunnelCopy, title = "DEBUG 10: temporal tunnel with valid range highlighted")


                # calculate the location of min and max value to find the best match
                min_valMID, max_valMID, min_locMID, max_locMID = cv2.minMaxLoc(resMID)

                if debugMode:
                    print("Template dimensions: H:{}xW:{}".format(h,w))
                    print('Dimension of the TT - H:{}xW:{}'.format(tunnel_h,tunnel_w))
                    print('Allowed temporal width: {}'.format(allowedW))
                    print('Allowed x-coords inside the TT: [{},{}]'.format(allowedW_xs,allowedW_xe))
                    print('Template matching result should have x-coords between [{},{}]'.format(validRangeOnTM[0],
                                                                                             validRangeOnTM[1]))
                    print('Actual x-coord of the max value: {}'.format(max_locMID[0]))

                if max_locMID[0]<validRangeOnTM[0] or max_locMID[0]>validRangeOnTM[1]: # invalid MID match!
                    if debugMode:
                        print('Outside temporal range of the TT!')
                    pass
                else:# valid MID match!
                    if debugMode:
                        print('Inside temporal range of the TT! Continue to the MSE analysis.')

                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_leftMID = min_locMID
                    else:
                        top_leftMID = max_locMID

                    # re-adjust to the coordinates of the original image
                    top_leftMID = (top_leftMID[0] + tunnel_topleft[0], top_leftMID[1] + tunnel_topleft[1])
                    # bottom_rightMID = (top_leftMID[0] + w, top_leftMID[1] + h)

                    matchMID = midIMG1C[top_leftMID[1]:top_leftMID[1] + h, top_leftMID[0]:top_leftMID[0] + w]

                    if debugMode:
                        whiteBar = np.zeros((h,3))
                        hstack = np.hstack((match, whiteBar))
                        hstack = np.hstack((hstack, template))
                        hstack = np.hstack((hstack, whiteBar))
                        hstack = np.hstack((hstack, matchMID))
                        showIMG(hstack,"DEBUG 11: BWD, MID, FWD MATCHES")

                    # matchTemplateSSIM = ssim(match,template)
                    # matchMIDTemplateSSIM = ssim(matchMID, mixed_template)
                    # matchTemplateMSE = mse(match, template)
                    matchMIDTemplateMSE = mse(matchMID, mixed_template)
                    MSERatio = matchMIDTemplateMSE/(w*h)

                    addr = "D:/detectron2/myProject/data/bi-GMM/test5/1/detectedSMV/"
                    cv2.imwrite(addr+str(idx)+"_merged.jpg",mixed_template)

                    if debugMode:
                        print('MSE MIXED/MID = {}'.format(int(matchMIDTemplateMSE), int(matchMIDTemplateMSE)))
                        print('MSE Ratio = {}'.format(MSERatio))
                        print('MSE Threshold: {}'.format(MSEthresh))

                    # if the patches are similar, or if the patches are big
                    # (bigger patches will likely be correctly classified in the next step)
                    if matchMIDTemplateMSE < MSEthresh: # if MSE is lower than the threshold (similar FWD, MID and BWD BBs)

                        result[counter-1][:, 0] = i
                        result[counter - 1][:, 1] = (top_leftMID[0], top_leftMID[1], w, h)
                        result[counter - 1][:, 2] = (top_left[0], top_left[1], w, h)

                        zerosMASK[zerosMASK == zerosMASK[top_left[1], top_left[0]]] = 0
                        counter += 1
                        # save results of the mid detection (there will always be one)

                    # print("Top left mid:{}, Bottom right mid:{}".format(top_leftMID,bottom_rightMID))

            else: # if max_val of the fwd/bwd template matching is zero, do not save the results
                pass

        # showTemplateMatchingResults(FWDimg, MIDimg, BWDimg, result)

    return result

def showTemplateMatchingResults(FWDimg, MIDimg, BWDimg, results, specific=0, filter=1):

    counter = 1

    if specific:
        # show a specific match, instead of all of them
        specific-=1

        randColor1 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        randColor2 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        randColor3 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # grab results for fwd and plot them
        xs, ys, w, h = results[specific, :, 0]
        cv2.rectangle(FWDimg, (xs, ys), (xs + w, ys + h), randColor1, 1)
        cv2.putText(FWDimg, str(specific+1) + "F", (xs - 5, ys - 5), 1, 1, (0, 255, 255), 1)

        # grab results for bwd and plot them
        top_left0, top_left1, w, h = results[specific, :, 2]
        bottom_right = (top_left0 + w, top_left1 + h)
        cv2.rectangle(BWDimg, (top_left0, top_left1), bottom_right, randColor2, 1)
        cv2.putText(BWDimg, str(specific+1) + "B", (top_left0 - 5, top_left1 - 5), 1, 1, (0, 255, 255), 1)

        # grab results for mid and plot them
        top_leftMID0, top_leftMID1, w, h = results[specific, :, 1]
        bottom_rightMID = (top_leftMID0 + w, top_leftMID1 + h)
        cv2.rectangle(MIDimg, (top_leftMID0, top_leftMID1), bottom_rightMID, randColor3, 1)
        cv2.putText(MIDimg, str(specific+1) + "M", (top_leftMID0 - 5, top_leftMID1 - 5), 1, 1, (0, 255, 255), 1)

    else:
        for i in range(0,results.shape[0]):
            # for each match,
            if sum(results[i,:,1]) == 0 and filter == 1:
                pass
            else:
                # only plot if the results includes a match for bwd and mid
                randColor = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                # grab results for fwd and plot them
                xs,ys,w,h = results[i,:,0]
                cv2.rectangle(FWDimg, (xs, ys), (xs + w, ys + h), randColor, 1)
                cv2.putText(FWDimg, str(counter) + "F", (xs - 5, ys - 5), 1, 1, (0, 255, 255), 1)

                # grab results for bwd and plot them
                top_left0, top_left1, w, h = results[i, :, 2]
                bottom_right = (top_left0 + w, top_left1 + h)
                cv2.rectangle(BWDimg, (top_left0,top_left1), bottom_right, randColor, 1)
                cv2.putText(BWDimg, str(counter) + "B", (top_left0 - 5, top_left1 - 5), 1, 1, (0, 255, 255), 1)

                # grab results for mid and plot them
                top_leftMID0, top_leftMID1, w, h = results[i, :, 1]
                bottom_rightMID = (top_leftMID0 + w, top_leftMID1 + h)
                cv2.rectangle(MIDimg, (top_leftMID0,top_leftMID1), bottom_rightMID, randColor, 1)
                cv2.putText(MIDimg, str(counter) + "M", (top_leftMID0 - 5, top_leftMID1 - 5), 1, 1, (0, 255, 255), 1)

                counter += 1

    # add the three images together
    blend = cv2.addWeighted(BWDimg, 0.5, FWDimg, 0.5, 0.0)
    blend = cv2.addWeighted(blend, 0.66, MIDimg, 0.33, 0.0)
    showIMG(blend, title="Template matching results with blended images")

def generateBlendResult(FWDimg, MIDimg, BWDimg, results, specific=0, filter=1):

    counter = 1

    blend = cv2.addWeighted(BWDimg, 0.5, FWDimg, 0.5, 0.0)
    blend = cv2.addWeighted(blend, 0.66, MIDimg, 0.33, 0.0)

    if specific:
        # show a specific match, instead of all of them
        specific-=1

        randColor1 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        randColor2 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        randColor3 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # grab results for fwd and plot them
        xs, ys, w, h = results[specific, :, 0]
        cv2.rectangle(blend, (xs, ys), (xs + w, ys + h), randColor1, 1)
        cv2.putText(blend, str(specific+1) + "F", (xs - 5, ys - 5), 1, 1, (0, 255, 255), 1)

        # grab results for bwd and plot them
        top_left0, top_left1, w, h = results[specific, :, 2]
        bottom_right = (top_left0 + w, top_left1 + h)
        cv2.rectangle(blend, (top_left0, top_left1), bottom_right, randColor2, 1)
        cv2.putText(blend, str(specific+1) + "B", (top_left0 - 5, top_left1 - 5), 1, 1, (0, 255, 255), 1)

        # grab results for mid and plot them
        top_leftMID0, top_leftMID1, w, h = results[specific, :, 1]
        bottom_rightMID = (top_leftMID0 + w, top_leftMID1 + h)
        cv2.rectangle(blend, (top_leftMID0, top_leftMID1), bottom_rightMID, randColor3, 1)
        cv2.putText(blend, str(specific+1) + "M", (top_leftMID0 - 5, top_leftMID1 - 5), 1, 1, (0, 255, 255), 1)

    else:
        for i in range(0,results.shape[0]):
            # for each match,
            if sum(results[i,:,1]) == 0 and filter == 1:
                pass
            else:
                # only plot if the results includes a match for bwd and mid
                randColor = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                # grab results for fwd and plot them
                xs,ys,w,h = results[i,:,0]
                cv2.rectangle(blend, (xs, ys), (xs + w, ys + h), randColor, 1)
                cv2.putText(blend, str(counter) + "F", (xs - 5, ys - 5), 1, 1, (0, 255, 255), 1)

                # grab results for bwd and plot them
                top_left0, top_left1, w, h = results[i, :, 2]
                bottom_right = (top_left0 + w, top_left1 + h)
                cv2.rectangle(blend, (top_left0,top_left1), bottom_right, randColor, 1)
                cv2.putText(blend, str(counter) + "B", (top_left0 - 5, top_left1 - 5), 1, 1, (0, 255, 255), 1)

                # grab results for mid and plot them
                top_leftMID0, top_leftMID1, w, h = results[i, :, 1]
                bottom_rightMID = (top_leftMID0 + w, top_leftMID1 + h)
                cv2.rectangle(blend, (top_leftMID0,top_leftMID1), bottom_rightMID, randColor, 1)
                cv2.putText(blend, str(counter) + "M", (top_leftMID0 - 5, top_leftMID1 - 5), 1, 1, (0, 255, 255), 1)

                counter += 1


    # add the three images together

    return blend

def ioUtwoBBs(boxA, boxB):

    # code by Adrian Rosebrock, 2016.
    # Available at: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # Accessed on May 28, 2020

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def classifyMultiplePatchesGMM(imgs, model, boatsThreshold = 2):
    # classify multiple images at a time based on a model
    # the input must be a NORMALIZED N C H W tensor
    if not (imgs.type() == 'torch.FloatTensor'):
        print('Incorrect data type. It must be a N C H W tensor of torch.FloatTensor type')
        exit(1)

    model.eval() # because only inference is going to be done
    with torch.no_grad():
        data = imgs.cuda()
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        pred = preds.data.cpu().numpy()


    for i in range(0,int(len(pred)/3)): # loop through each group of 3 BBs
        sum = pred[i*3]+pred[i*3+1]+pred[i*3+2]
        if sum < boatsThreshold: # if the number of boats identified in the group is less than the threshold, ignore it
            pred[i * 3] = pred[i*3+1] = pred[i*3+2] = 0

    return pred

def mergeODandDSMV(objDet, scoresDet, biGMM):

    nBBbiGMM = len(biGMM) # number of BBs. Note that some can be zero (when the BBs were classified as background)

    for i in range(0,nBBbiGMM): # for each BB in the biGMM

        if biGMM[i] is not 0:
            nValidVessels = biGMM[i].shape[1] # number of valid vessels in each stage (B M F)
            for j in range(nValidVessels): # for each valid BB in a stage,
                currentBB = biGMM[i][:,j]
                stage = currentBB[4] # either 0:BWD, 1:MID or 2:FWD

                # add this bb to the right position in the obj. det. results
                objDet[stage].append(np.array([currentBB[0], currentBB[1], currentBB[0] + currentBB[2], currentBB[1] + currentBB[3]]))
                scoresDet[stage].append(0)

    return objDet, scoresDet

def concatObjDetAndBiGMM(blendedResult, blendedScore):
    concatBBs = []
    concatScores = []

    for i in range(0,3):

        # print('now {} and {}'.format(blendedResult[i], blendedScore[i]))
        cBBs, cScores, cClasses = concatenateBBs(blendedResult[i], blendedScore[i])
        concatBBs.append(cBBs)
        concatScores.append(cScores)

    return concatBBs, concatScores

def filterDSMVresults(groups_of_three, result, blend, outDirectory, images, transform, model, thresh):

    # change the indexes so the rest of the program does not need to be changed
    for idx,i in enumerate(result):
        safe = i.copy()
        i[:, 0] = safe[:, 2]
        i[:, 2] = safe[:, 0]

    gt = groups_of_three
    cv2.imwrite(outDirectory + gt["uniqueIDMinute"] + "_.jpg", blend)  # first save the blended image
    onlyVessels = 0

    patchesList = []  # create a list of to store all the BBs' patches of this group of images.

    for bbox in range(len(result)):
        for j in range(result[bbox].shape[1]):  # loop on each group of 3 bounding boxes (F M B)
            # j = 0: backward img, j=1: mid img, j=2: forward img
            BB = result[bbox][:, j]
            patch = images[j][BB[1]:BB[1] + BB[3], BB[0]:BB[0] + BB[2]]  # grab a patch on the image
            patch = cv2.resize(patch, (224, 224))  # resize it to 224,224
            patchesList.append(transform(patch))  # append it to a list
            # showIMG(patch)
            cv2.imwrite(outDirectory + gt["uniqueIDMinute"] + "_" + str(bbox + 1) + "_" + str(j) + ".jpg", patch)
            # create an image of the patch and save it

    filteredBBs = [0]*len(result) # create a list for the filtered results

    # classify all the patches using the pre-trained model
    if len(patchesList) > 0:

        tensor = torch.stack(patchesList)  # turn the list of patches into a N C H W torch tensor

        # classify the patches using the image classifying model. they are already normalized tensors
        preds = classifyMultiplePatchesGMM(tensor, model, boatsThreshold=thresh)

        # TEST ARRAY. COMMENT IT WHEN RUNNING THE PROGRAM. DEBUG ONLY.
        # preds = [1,0,0,1,0,1,0,1,0]

        # delete the bounding boxes not classified as vessels
        onlyVessels = result.copy()
        for i in range(len(result)):
            # print('BOUNDING BOX {}'.format(i))
            currentGroupBB = result[i]
            currentGroupPred = preds[i*3:(i*3)+3]

            if sum(currentGroupPred) > 0: # if this list of 3 BBs contains any valid detection,
                newValue = []
                for predIdx in range(0,3): # move through each detection idx (0,1,2)

                    if currentGroupPred[predIdx]: # if the current detection is a vessel (1) ano not bkgd (0),
                        valueToAdd = np.append(currentGroupBB[:,predIdx], predIdx) # add 0,1 or 2 to the end of the
                        # value so that we know later on if this is a fwd, bwd or mid bb
                        newValue.append(valueToAdd)
                        array = np.vstack(newValue) # combine all list elements into an array
                        onlyVessels[i] = np.transpose(array) # transpose so that each column represents a valid biGMM BB

            else: # if you have a (0 0 0) prediction, ignore this group of bounding boxes
                onlyVessels[i] = 0

        name = outDirectory + gt["uniqueIDMinute"] + "_GMMClassOUT.jpg"
        plotClassificationResults(images, preds, result, name)

    return onlyVessels

def createGroupsofThree(files,inDirectory,prefix):
    divided_files = []
    groups_of_three = []
    indexGT = 0

    # first, read the name of all files and specify date and time in a dictionary
    for i in range(len(files)):
        filePath = inDirectory + files[i]
        parts = files[i][len(prefix):-4]
        uniqueIDMinute = parts[:-3]

        # for the first group, create a new dictionary
        if len(groups_of_three) == 0:
            groups_of_three.append({"uniqueIDMinute": uniqueIDMinute})
            groups_of_three[-1]["image1"]=filePath
        else:

            # check if this minute group is already in the list
            index = next((index for index, item in enumerate(groups_of_three) if item["uniqueIDMinute"] == uniqueIDMinute), "No")
            # print('looking for {} found? {}'.format(uniqueIDMinute, index))

            if index == "No":
                groups_of_three.append({"uniqueIDMinute": uniqueIDMinute})
                groups_of_three[-1]["image1"] = filePath
            else:
                variablename = "image"+str(len(groups_of_three[index]))
                groups_of_three[index][variablename] = filePath

    return  groups_of_three


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
