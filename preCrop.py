import os
from PIL import Image

def crop_8x8(image_name, image_path, gt_path):
    image = Image.open(image_path)
    gt = Image.open(gt_path)

    box = [200, 200, 1000, 1000]

    roi_image = image.crop(box)
    roi_gt = gt.crop(box)

    img_tailor = os.path.join(os.getcwd(), 'images')

    destination_image = os.path.join(img_tailor, "Original\\" + image_name + ".tif")
    destination_gt = os.path.join(img_tailor, "GroundTruth\\" + image_name + ".tif")
    roi_image.save(destination_image)
    roi_gt.save(destination_gt)

if __name__ == '__main__':
    image_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011']
    for item in image_list:
        image_path = os.path.join(os.getcwd(), 'tmp\\Original\\' + item + ".tif")
        gt_path = os.path.join(os.getcwd(), 'tmp\\GroundTruth\\' + item + ".tif")
        crop_8x8(item, image_path, gt_path)
    print("done!")


