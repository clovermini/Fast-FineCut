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

# 测试算法，使用品质因素输出算法准确度
if __name__ == "__main__":
    ROOT_DIR = os.path.join(os.getcwd(), "images")
    lastSegmentImageAddress = os.path.join(ROOT_DIR, "GroundTruth")   # 上一层图片分割结果所在地址
    nextOriginalImageAddress = os.path.join(ROOT_DIR, "Original")  # 本层待分割原图所在地址
    nextSegmentAddress = os.path.join(ROOT_DIR, "Results/Waggoner")   # 本层待分割图片分割结果存放地址
    # index = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']   # 图片编号列表
    index = ['001', '002', '003', '004']  # 图片编号列表

    T = 0
    Length = 0
    F = 0

    for i in range(1, len(index)):
        Length += 1
        print("image : ", index[i])

        last_Label = lastSegmentImageAddress + "/" + index[i - 1] + '.tif'  # 上层人工分割结果
        next_Label = lastSegmentImageAddress + "/" + index[i] + '.tif'    # 本层人工分割结果
        next_Original = nextOriginalImageAddress + "/" + index[i] + '.tif'  # 本层原图
        next_Segment = nextSegmentAddress + "/" + index[i] + '.tif'  # 分割结果存放地址

        # 读取并细化人工分割结果
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
        # Fast-FineCut 算法   若使用 waggoner 算法, 设置 algorithm="wag"
        segment = cropAndPaste.segment(200, 200, last_Label, next_Original, next_Segment, algorithm="wag")
        img_root = os.path.join(os.getcwd(), 'tmp')
        rmtree(img_root)
        os.mkdir(img_root)

        segmentTemp = cv2.imread(next_Segment)
        segmentResult = np.zeros((rowNumber, colNumber))
        if segmentTemp.ndim == 3:
            segmentResult = cv2.cvtColor(segmentTemp, cv2.COLOR_BGR2GRAY)
        elif segmentTemp.ndim == 2:
            segmentResult = segmentTemp

        # 传播分割算法(不使用 Local propagation method based on Overlap-tile strategy), 若使用waggoner算法, 设置algorithm="wag"
        # segmentResult = propagationSegment(last_Label, next_Original, next_Segment, algorithm="ffc")
        end_time = time.time()

        # 细化分割结果
        segmentResult_fc = morphology.skeletonize(segmentResult / 255) * 255

        cv2.imwrite(next_Segment, segmentResult_fc)
        F_test = evaluateMeritForEdge(segmentResult, nLab_edge_fc)
        print("name = " + index[i] + " , F = " + str(F_test) + ' , time = ' + str(end_time - start_time))

        F += F_test
        T += (end_time - start_time)

    print("average F = " + str(F / Length) + " , average T = " + str(T / Length))

