import os
from Algorithm.propagation import propagationSegment
from PIL import Image

# Image cropping and stitching
# Each image is first cropped into (sizeX, sizeY) image, then each image is segmented by graph-cut. Finally, the segmented image is re-stitched.
# The cropping order is horizontal first and then vertical
# tmp Temporary address of intermediate results

# cropping
def tailor(sizeX, sizeY, lastAddress, nextAddress, saveAddress, region = 10):
    image_last = Image.open(lastAddress)
    image_next = Image.open(nextAddress)

    if image_last.size != image_next.size:
        return "Error! The image of the previous layer and the image of the layer is different in dimension."

    (w_image, h_image) = image_last.size

    row = int(h_image/sizeY)
    col = int(w_image/sizeX)

    n = 0
    for i in range(row):
        for j in range(col):
            box_X = j*sizeX
            box_Y = i*sizeY
            box_W = sizeX
            box_H = sizeY
            if i == 0:
                box_Y = 0
                box_H = sizeY + region
            elif i == row-1:
                box_Y = i*sizeY - region
                box_H = sizeY + region
            else:
                box_Y = i*sizeY - region
                box_H = sizeY + 2*region
            if j == 0:
                box_X = 0
                box_W = sizeX + region
            elif j == col-1:
                box_X = j*sizeX - region
                box_W = sizeX + region
            else:
                box_X = j*sizeX - region
                box_W = sizeX + 2*region
            box = (box_X, box_Y, box_X + box_W, box_Y + box_H)
            roi_last = image_last.crop(box)
            roi_next = image_next.crop(box)
            n += 1
            try:
                destination_last = os.path.join(saveAddress, "last"+str(i)+"-"+str(j)+'.tif')
                destination_next = os.path.join(saveAddress, "next"+str(i)+"-"+str(j)+'.tif')
                roi_last.save(destination_last)
                roi_next.save(destination_next)
            except:
                return "Picture is saved incorrectly"
    return "success"

# Get the maximum number of rows and columns for a cropped picture (saved with row-column names)
def getXAndY(pieceAddress):
    imgList = os.listdir(pieceAddress)
    tmpList = []
    tmpX = 0    # row
    tmpY = 0    # column
    for img in imgList:
        tmpList.append(img.replace('result', '').replace('last', '').replace('next', '').replace('.tif', '').split('-'))
    for size in tmpList:
        if int(size[0]) > tmpX:
            tmpX = int(size[0])
        if int(size[1]) > tmpY:
            tmpY = int(size[1])
    return [tmpX, tmpY]

# stitching
def stitch(sizeX, sizeY, pieceAddress, saveRoad, region = 10):
    tmpX = getXAndY(pieceAddress)[0]
    tmpY = getXAndY(pieceAddress)[1]
    resultWidth = sizeX*(tmpY+1)    # width
    resultHeight = sizeY*(tmpX+1)   # height
    result = Image.new("RGB", (resultWidth, resultHeight))
    for i in range(tmpX+1):
        for j in range(tmpY+1):
            fname = os.path.join(pieceAddress, "result"+str(i)+"-"+str(j)+'.tif')
            piece = Image.open(fname)
            box_H = sizeY
            box_W = sizeX
            if i == 0:
                box_Y = 0
            else:
                box_Y = region
            if j == 0:
                box_X = 0
            else:
                box_X = region
            box = (box_X, box_Y, box_X + box_W, box_Y + box_H)
            roi_result = piece.crop(box)
            result.paste(roi_result, (j*sizeX, i*sizeY))
    result.save(saveRoad)
    return result


def segment(sizeX, sizeY, lastLabeled, nextOriginal, nextSegment, algorithm = "ffc", boxLength = 18, boundingLength = 18, KCost = 3):
    """
    Function：The main function of the Fast-FineCut boundary detection algorithm, including image cropping, propagation segmentation, stitching and so on
        Input：sizeX: the width of the crop picture, recommended value 200
             sizeY: the height of the crop picture, recommended value 200
             lastLabeled: segmentation result of last image
             nextOriginal: this layer's original image
             nextSegment: storage address of this layer's segmentation result
             algorithm: Algorithm category Fast-FineCut -- "ffc"  Waggoner -- "wag"
             boxLength: Overlap area, same as BoundingLength by default
             boundingLength: the length of bounding region, default 18
             infiniteCost: Default 100, refers to infinity
             KCost: K value of binary term in propagation segmentation, default 3
        Output: result：Segmentation result, it should be noted that the result is opened by Image, not opencv
    """

    # Temporary address of cropping picture, default in the directory where this project located
    img_tailor = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(img_tailor):
        os.mkdir(img_tailor)

    if os.path.exists(img_tailor) and os.path.exists(lastLabeled) and os.path.exists(nextOriginal):
        tailor(sizeX,sizeY,lastLabeled, nextOriginal,img_tailor, boxLength)
        [tmpX, tmpY] = getXAndY(img_tailor)
        for i in range(tmpX+1):
            for j in range(tmpY+1):
                lastSegmentImageAddress = os.path.join(img_tailor, "last"+str(i)+"-"+str(j)+'.tif')
                nextOriginalImageAddress = os.path.join(img_tailor, "next"+str(i)+"-"+str(j)+'.tif')
                nextSegmentAddress = os.path.join(img_tailor, "result"+str(i)+"-"+str(j)+'.tif')
                propagationSegment(lastSegmentImageAddress, nextOriginalImageAddress, nextSegmentAddress, algorithm = algorithm, boundingLength=boundingLength, KCost=KCost)
        result = stitch(sizeX, sizeY, img_tailor, nextSegment, boxLength)
        return result
    else:
        return "Image address error"
