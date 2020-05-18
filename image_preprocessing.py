# -*- coding:utf-8 -*-
'''
this script is used for basic process of lung 2017 in Data Science Bowl
'''
import cv2
import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import os


#判断某文件是否是mhd格式的文件
def is_mhd_file(filepath):
    file=os.path.splitext(filepath)
    type=file[1]
    if type == '.mhd' or type == '.MHD':
        return True
    return False

#获取原始图像像素值（-4000,4000）
def get_pixels_hu_by_simpleitk(mhd_dir):
    image=sitk.ReadImage(mhd_dir)
    im_array=sitk.GetArrayFromImage(image)
    return im_array

#图像像素归一化
def normalize_hu(im):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image=im  #不改变原始数组
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

#肺实质分割
# numpyImage[numpyImage > -600] = 1
# numpyImage[numpyImage <= -600] = 0
def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -600
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
 
    plt.show()
 
    return im

##保存肺实质分割图像
def save_data(numpyImage,save_dir):
    ####nupyImage为原始三维图像,save_dir为保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(numpyImage.shape[0]):
        im=get_segmented_lungs(numpyImage[i],plot=False)
        image=normalize_hu(im)
        cv2.imwrite(save_dir+'/'+str(i)+'.png',image*255)

    print(save_dir)

 
if __name__ == '__main__':
    src_dir='./data/raw'
    files=os.listdir(src_dir)
    for s in files:
        filepath=src_dir+'/'+s
        if is_mhd_file(filepath):
            numpyImage=get_pixels_hu_by_simpleitk(filepath)
            save_dir='./data/segmenteddata/'+os.path.splitext(s)[0]
            save_data(numpyImage,save_dir)
    
    print("--------imwrite over----------")