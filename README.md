# Fast-FineCut
This is an implementation of Fast-FineCut algorithm in Python 3, which is a boundary detection algorithm in microscopic images considering 3D information.

# Requirements
Python 3 need to be installed before running this scripts.
To run this algorithm, you need to install the python packages as follows:
  
    opencv 3
    gco(gco_python-master, gco_v3.0)

To install gco, you need to download the `gco_python-master` and `gco_v3.0` at [here](https://github.com/clovermini/Fast-FineCut/releases/tag/v1.0) first, after download, enter the  `gco_python-master` directory, open the setup file and change the gco_directory to your `gco_v3.0` directory, and then, run the command:

    python setup.py install

# DataSet and Running
We provide 11 test microscopic images in the folder `images`, including the `Original`, `GroundTruth` and `Results`, the `Results` includes the detection result of Fast-FineCut algorithm, Wagonner's algorithm and Others. 

If you want to run the Fast-FineCut algorithm and Waggoner's algorithm, run the `fastFineCutDemo.py`, if you want to run the other algorithm, including `Otsu`, `Iteractive`, `Canny` and `Adaptive Threshold`, run the `edgeExtractAllDemo.py`, and if you want to run this on your own datasets, you have to change the corresponding image address in those file.

The example results of Fast-FineCut algorithm is shown as follows: 
![](https://raw.githubusercontent.com/clovermini/MarkdownPhotos/master/004.png)


