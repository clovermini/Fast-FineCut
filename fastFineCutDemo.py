from skimage import morphology
import Algorithm.cropAndPaste as cropAndPaste
from Algorithm.propagation import propagationSegment
import os
import numpy as np
import cv2
import time
from Algorithm.propagationLabel import getEdgesFromLabel,evaluateMeritForEdge
from skimage.measure import label
from shutil import rmtree

# Fast-Fine Cut Algorithm Demo, we use the figure of merit to measure the accuracy of algorithm
if __name__ == "__main__":
    ROOT_DIR = os.path.join(os.getcwd(), "images")
    lastSegmentImageAddress = os.path.join(ROOT_DIR, "GroundTruth")   # The address of segment result of last image
    nextOriginalImageAddress = os.path.join(ROOT_DIR, "Original")     # The address of the original image to be segmented in this layer
    nextSegmentAddress = os.path.join(ROOT_DIR, "Results/Waggoner")   # The storage address of segmentation results
    index = ['001', '002', '003', '004', '005']  # List of image_ids

    T = 0
    Length = 0
    F = 0

    for i in range(1, len(index)):
        Length += 1
        print("segment image : ", index[i])

        last_Label = lastSegmentImageAddress + "/" + index[i - 1] + '.tif'  # Ground truth image of last image
        next_Label = lastSegmentImageAddress + "/" + index[i] + '.tif'    # Ground truth image of this image
        next_Original = nextOriginalImageAddress + "/" + index[i] + '.tif'  # This layer's original image
        next_Segment = nextSegmentAddress + "/" + index[i] + '.tif'  # Storage path of segmentation result

        # read and skeletonize this layer's ground truth image
        nLab = cv2.imread(next_Label)
        rowNumber = nLab.shape[0]
        colNumber = nLab.shape[1]
        nSegment = np.zeros((rowNumber, colNumber))
        nSegment = cv2.cvtColor(nLab, cv2.COLOR_BGR2GRAY)
        (nLabeled, nNumber) = label(nSegment, neighbors=4, return_num=True)
        nLabeled = nLabeled.astype(np.int32)
        nLab_edge = np.zeros((rowNumber, colNumber))
        nLab_edge = getEdgesFromLabel(nLabeled)
        nLab_edge_fc = morphology.skeletonize(nLab_edge / 255) * 255

        start_time = time.time()
        # Fast-FineCut algorithm , if want to use waggoner's algorithm, set the parameter algorithm="wag"
        # note!!! The size of should be the times of (sizeX, sizeY)
        segment = cropAndPaste.segment(sizeX=200, sizeY=200, lastLabeled=last_Label, nextOriginal=next_Original, nextSegment=next_Segment, algorithm="ffc")
        img_root = os.path.join(os.getcwd(), 'tmp')
        rmtree(img_root)
        os.mkdir(img_root)

        segmentTemp = cv2.imread(next_Segment)
        segmentResult = np.zeros((rowNumber, colNumber))
        if segmentTemp.ndim == 3:
            segmentResult = cv2.cvtColor(segmentTemp, cv2.COLOR_BGR2GRAY)
        elif segmentTemp.ndim == 2:
            segmentResult = segmentTemp

        # propagation segmentation algorithm ( without Local propagation method based on Overlap-tile strategy)
        # if use the waggoner algorithm, please set the parameter algorithm="wag"
        # segmentResult = propagationSegment(last_Label, next_Original, next_Segment, algorithm="ffc")
        end_time = time.time()

        # Skeletonization
        segmentResult_fc = morphology.skeletonize(segmentResult / 255) * 255

        cv2.imwrite(next_Segment, segmentResult_fc)
        F_test = evaluateMeritForEdge(segmentResult, nLab_edge_fc)
        print("name = " + index[i] + " , F = " + str(F_test) + ' , time = ' + str(end_time - start_time))

        F += F_test
        T += (end_time - start_time)

    print("average F = " + str(F / Length) + " , average T = " + str(T / Length))

