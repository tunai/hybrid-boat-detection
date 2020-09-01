import cv2
import os
import numpy as np
from utils import filterDetection, pickModelParameters, CCCalculateandFilter, templateMatching, \
    showTemplateMatchingResults, generateBlendResult
from utils_plotting import showIMG, plotAllBB, plotBB_BS_Result
import timeit
import time

def bidirectionalGMM(bwdImg, midImg, fwdImg,
                     pixelThresh =10,
                     pixelDeltaThresh = 120,
                     validRange = [220,500],
                     displayResults = None,
                     displayAllBB = None,
                     outputBlendImg = None,
                     upperBBlimit = 6,
                     debugMode = False):

    # inputs:
    # pixelThresh = 10  # pixel size threshold used in the morphological operations
    # pixelDeltaThresh = 110  # pixel intensity change threshold (for the background subtraction process)
    # validRange = [220,500] # range of values in the y-axis of the image to be considered

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # create an ellipse morphological element
    kernelOpening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    files = [bwdImg,midImg,fwdImg]
    imageMID = files[1]
    imageMID = imageMID[validRange[0]:validRange[1], :]

    numFiles = 3

    #start_time = time.time()

    numCC = upperBBlimit + 1
    firstRun = True
    deltaThresh = pixelDeltaThresh # starts with user-provided (or default) pixel delta threshold

    while numCC > upperBBlimit: # guarantee that the background subtraction is done until a number lower than the
        # upperBBLimit of regions is found.

        # eliminate the patches with a mean intensity higher than a set threshold

        # print('Creating background models with pixel threshold of {}'.format(deltaThresh))
        fgbgFORWARD = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=deltaThresh, detectShadows=False)
        fgbgBACKWARD = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=deltaThresh, detectShadows=False)
        deltaThresh += 100

        for i in range(0, numFiles):

            imageFORWARD = files[i]
            imageBACKWARD = files[numFiles - i - 1]
            imageFORWARD = imageFORWARD[validRange[0]:validRange[1], :]  # eliminate the clock, sky from the images
            imageBACKWARD = imageBACKWARD[validRange[0]:validRange[1],:]  # eliminate the clock, sky from the images

            fgmaskFORWARD = fgbgFORWARD.apply(imageFORWARD)
            fgmaskBACKWARD = fgbgBACKWARD.apply(imageBACKWARD)

            # showIMG(fgmaskFORWARD, title = "FGMASK before filtering")
            # showIMG(fgmaskBACKWARD, title = "BGMASK before filtering")

            fgmaskFORWARD = cv2.morphologyEx(fgmaskFORWARD, cv2.MORPH_OPEN, kernel)
            fgmaskBACKWARD = cv2.morphologyEx(fgmaskBACKWARD, cv2.MORPH_OPEN, kernel)

        fgmaskFORWARD = cv2.morphologyEx(fgmaskFORWARD, cv2.MORPH_DILATE, kernelOpening)
        fgmaskBACKWARD = cv2.morphologyEx(fgmaskBACKWARD, cv2.MORPH_DILATE, kernelOpening)

        ccBB_FORWARD = CCCalculateandFilter(imageFORWARD, fgmaskFORWARD, pixelThresh, 0)

        if np.isscalar(ccBB_FORWARD):
            print("No fwd CC! Exiting function...")
            return 0, 0 if (outputBlendImg is not None) else 0

        if firstRun is False: # if after the first run you are still over the limit of ccBB, check for the intensity of
            # each ccBB to try and exclude the ones with high mean intensity (usually the sunlight's reflection)

            # print('got here')
            dummy = fwdImg[validRange[0]:validRange[1], :].copy()
            resultCC = []
            for idx, bbNow in enumerate(ccBB_FORWARD):
                patch = dummy[bbNow[1]:bbNow[1]+bbNow[3], bbNow[0]:bbNow[0]+bbNow[2], :]
                if patch.mean()<145:
                    # print('wooow')
                    resultCC.append(ccBB_FORWARD[idx])
                    # cv2.rectangle(dummy, (bbNow[0], bbNow[1]), (bbNow[0] + bbNow[2], bbNow[1] + bbNow[3]), (255, 255, 255), 1)

            ccBB_FORWARD = np.array(resultCC)
            # showIMG(dummy)

        numCC = ccBB_FORWARD.shape[0]
        # print(numCC)
        firstRun = False
        # print('Number of fwd direction motion BBs: {}'.format(numCC))

    if np.isscalar(ccBB_FORWARD):
        print("No fwd CC! Exiting function...")
        return 0,0 if (outputBlendImg is not None) else 0

    ccBB_BACKWARD = CCCalculateandFilter(imageBACKWARD, fgmaskBACKWARD, pixelThresh, 0)
    if np.isscalar(ccBB_BACKWARD):
        print("No bwd CC! Exiting function...")
        return 0,0 if (outputBlendImg is not None) else 0

    TMresult = templateMatching(imageFORWARD, imageMID, imageBACKWARD, ccBB_FORWARD, ccBB_BACKWARD, debugMode=debugMode)

    filteredTMResult = []
    for i in range(len(TMresult)):
        # print(sum(TMresult[i, :, 1]))
        if sum(TMresult[i, :, 1]) == 0:
            pass
        else:
            # print('add this one')
            temp = TMresult[i, :, :].copy()
            temp[1,:] = temp[1,:] + validRange[0] # re-reference the coordinates to the original ones
            # to do that, simply add validRange[0] to the initial y coordinates of all BBs.
            filteredTMResult.append(temp)

    # print("detection time = {} seconds".format(time.time() - start_time))

    if displayAllBB is not None:
        ccBBBWD = []
        ccBBFWD = []

        for i in range(0,len(ccBB_BACKWARD)):
            xi = ccBB_BACKWARD[i][0]
            yi = ccBB_BACKWARD[i][1]
            xf = ccBB_BACKWARD[i][0] + ccBB_BACKWARD[i][2]
            yf = ccBB_BACKWARD[i][1] + ccBB_BACKWARD[i][3]
            ccBBBWD.append([xi,yi,xf,yf])

        for i in range(0,len(ccBB_FORWARD)):
            xi = ccBB_FORWARD[i][0]
            yi = ccBB_FORWARD[i][1]
            xf = ccBB_FORWARD[i][0] + ccBB_FORWARD[i][2]
            yf = ccBB_FORWARD[i][1] + ccBB_FORWARD[i][3]
            ccBBFWD.append([xi,yi,xf,yf])

        fwdFinal = plotBB_BS_Result(imageFORWARD, ccBBFWD, color=(0, 255, 255), line=1, id="F")
        bwdFinal = plotBB_BS_Result(imageBACKWARD, ccBBBWD, color=(0, 255, 255), line=1, id="B")
        blend = cv2.addWeighted(fwdFinal, 0.5, bwdFinal, 0.5, 0.0)
        showIMG(blend)

    if displayResults:
        showTemplateMatchingResults(imageFORWARD.copy(), imageMID.copy(), imageBACKWARD.copy(),
                                    TMresult, specific=0, filter=1)

    if outputBlendImg is True:
        return filteredTMResult, generateBlendResult(imageFORWARD.copy(), imageMID.copy(), imageBACKWARD.copy(),
                                                    TMresult, specific=0, filter=1)
    else:
        return filteredTMResult
