# Fast-FineCut
## Citation
This is an implementation of Fast-FineCut algorithm in Python 3, which is a boundary detection algorithm in microscopic images considering 3D information. 

If you use it successfully for your research please be so kind to cite our work:

<br/>@article{article,
<br/>author = {Ma, Boyuan and Ban, Xiaojuan and Su, Ya and Liu, Chuni and Wang, Hao and Xue, Weihua and Zhi, Yonghong and Wu, Di},
<br/>year = {2018},
<br/>month = {09},
<br/>pages = {},
<br/>title = {Fast-FineCut: Grain boundary detection in microscopic images considering 3D information},
<br/>booktitle = {Micron}
<br/>}

or

Ma B, Ban X, Su Y, Liu C, Wang H, Xue W. "Fast-FineCut : Boundary detection in microscopic images considering 3D information", DOI: 10.1016/j.micron.2018.09.002

## Introduction
The inner structure of a material is called microstructure. It stores the genesis of a material and determines all its physical and chemical properties. However, it is still a big challenge to detect key information in microscopic images fastly
and accuratly. In this work, we address the task of grain boundary detection in microscopic image processing and develop a graph-cut based method called Fast-FineCut to solve the problem. Our algorithm makes two key contributions: 1) an improved approach that incorporates 3D information between slices as domain knowledge, which can detect the boundaries precisely, even for the vague and missing boundaries. 2) a local processing method based on overlap-tile strategy, which can not only solve the "chain scission" problem at the edge of images, but save the consumption of resources (such as computational time and memory space), making it possible to analyze the microscopic images with huge resolution.


## Requirements
Python 3 need to be installed before running this scripts.
To run this algorithm, you need to install the python packages as follows:

    opencv 3
    gco(gco_python-master, gco-v3.0)

To install gco, you need to download the `gco_python-master` and `gco-v3.0` at [here](https://github.com/clovermini/Fast-FineCut/releases/tag/v1.0) first, after download, enter the  `gco_python-master` directory, open the setup file and change the gco_directory to your `gco-v3.0` directory, and then, run the command:

    python setup.py install

## DataSet and Running
We provide 5 test microscopic images in the folder `images`, including the `Original`, `GroundTruth` and `Results`, the `Results` includes the detection result of Fast-FineCut algorithm, Wagonner's algorithm and Others. 

If you want to run the Fast-FineCut algorithm and Waggoner's algorithm, run the `fastFineCutDemo.py`, if you want to run the other algorithm, including `Otsu`, `Iteractive`, `Canny` and `Adaptive Threshold`, run the `edgeExtractAllDemo.py`, and if you want to run this on your own datasets, you have to change the corresponding image address in those files.

The example results of Fast-FineCut algorithm is shown as follows: 

<p align = "center">
<img src="https://raw.githubusercontent.com/clovermini/MarkdownPhotos/master/005.png">
</p>
