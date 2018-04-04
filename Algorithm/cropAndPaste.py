import os
from Algorithm.propagation import propagationSegment
from PIL import Image

# 图像的裁剪和拼接
# 每一张图像首先裁剪成200*200大小的图像，分别进行图割，最后将分割后的图像重新拼接
# 裁剪顺序为先横向再纵向
# tmp 裁剪图片暂存地址

# 裁剪
def tailor(sizeX, sizeY, lastAddress, nextAddress, saveAddress, region = 10):
    image_last = Image.open(lastAddress)
    image_next = Image.open(nextAddress)

    if image_last.size != image_next.size:
        return "上一层图像和本层图像维度不一样，错误"

    (w_image, h_image) = image_last.size

    row = int(h_image/sizeY)
    col = int(w_image/sizeX)

    # 裁剪成200*200的图片
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
                return  "图片保存错误"
    return "success"

# 获取裁剪图片（以行-列名称保存）的最大行数和列数
def getXAndY(pieceAddress):
    imgList = os.listdir(pieceAddress)
    tmpList = []
    tmpX = 0    # 行
    tmpY = 0    # 列
    for img in imgList:
        tmpList.append(img.replace('result', '').replace('last', '').replace('next', '').replace('.tif', '').split('-'))
    for size in tmpList:
        if int(size[0]) > tmpX:
            tmpX = int(size[0])
        if int(size[1]) > tmpY:
            tmpY = int(size[1])
    return [tmpX, tmpY]

# 拼接
def stitch(sizeX, sizeY, pieceAddress, saveRoad, region = 10):
    # 拼接图片
    tmpX = getXAndY(pieceAddress)[0]
    tmpY = getXAndY(pieceAddress)[1]
    resultWidth = sizeX*(tmpY+1)    # 宽
    resultHeight = sizeY*(tmpX+1)   # 高
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

'''
图像分割，包含图像裁剪、图割、去毛刺和拼接一系列操作
sizeX  裁剪图片宽
sizeY  裁剪图片高
lastLabeled  上一层已分割图像地址
nextOriginal  本层原始图像地址
nextSegment  本层分割结果保存地址
finLength  去除毛刺阈值,默认为8
boundingLength  默认16，边界区域大小
infiniteCost   默认100，指无穷大
'''
def segment(sizeX, sizeY, lastLabeled, nextOriginal, nextSegment, algorithm = "ffc", boxLength = 18, boundingLength = 18, KCost = 3):
    """
    功能：Fast-FineCut边界提取算法主函数，包含图像裁剪、传播分割、拼接一系列操作
        输入：sizeX  裁剪图片宽，推荐值200
             sizeY  裁剪图片高，推荐值200
             lastLabeled  上一层已分割图像地址
             nextOriginal  本层原始图像地址
             nextSegment  本层分割结果保存地址
             algorithm  算法类别 Fast-FineCut -- "ffc"  Waggoner -- "wag"
             boxLength  重叠区域，默认与BoundingLength相同
             boundingLength 边界区域长度，默认18
             infiniteCost   默认100，指无穷大
             KCost 传播分割中二元项参数，默认为3
        输出: result：分割结果，需要注意的是结果由Image打开，并非opencv打开
              bounndingRegion：属于该region像素值为255，不属于为0（可选）
    """

    # 裁剪图片暂存地址, 默认位于项目所在目录
    img_tailor = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(img_tailor):
        os.mkdir(img_tailor)

    if os.path.exists(img_tailor) and os.path.exists(lastLabeled) and os.path.exists(nextOriginal):
        tailor(sizeX,sizeY,lastLabeled, nextOriginal,img_tailor, boxLength)
        [tmpX, tmpY] = getXAndY(img_tailor)
        for i in range(tmpX+1):
            for j in range(tmpY+1):
                print("segment >>> ", i, "---", j)
                lastSegmentImageAddress = os.path.join(img_tailor, "last"+str(i)+"-"+str(j)+'.tif')
                nextOriginalImageAddress = os.path.join(img_tailor, "next"+str(i)+"-"+str(j)+'.tif')
                nextSegmentAddress = os.path.join(img_tailor, "result"+str(i)+"-"+str(j)+'.tif')
                propagationSegment(lastSegmentImageAddress, nextOriginalImageAddress, nextSegmentAddress, algorithm = algorithm, boundingLength=boundingLength, KCost=KCost)
        result = stitch(sizeX, sizeY, img_tailor, nextSegment, boxLength)
        return result
    else:
        return "图像地址错误"
