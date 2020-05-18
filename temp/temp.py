import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from scipy import ndimage as ndi

def normalize_hu(im):
    image = cv2.normalize(im, None, 0, 255)
    return image

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
        #plt.savefig('./temp/binary.png')
        #print(binary)
        cv2.imwrite("./temp/binary.png",(binary+0)*255)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
        #plt.savefig('./temp/cleared.png')
        cv2.imwrite("./temp/cleared.png",(cleared+0)*255)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
        #plt.savefig('./temp/label.png')
        cv2.imwrite("./temp/label.png",(label_image+0)*255)
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
        #plt.savefig('./temp/binary2.png')
        cv2.imwrite("./temp/binary2.png",(binary+0)*255)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
        #plt.savefig('./temp/binary3.png')
        cv2.imwrite("./temp/binary3.png",(binary+0)*255)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
        #plt.savefig('./temp/binary4.png')
        cv2.imwrite("./temp/binary4.png",(binary+0)*255)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
        #plt.savefig('./temp/binary4.png')
        cv2.imwrite("./temp/binary5.png",(binary+0)*255)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
        #plt.savefig('./temp/im.png')
        cv2.imwrite("./temp/im.png",(im+0)*255)
 
    plt.show()
 
    return im


mhd_dir = "./temp/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd"
image=sitk.ReadImage(mhd_dir)
im_array=sitk.GetArrayFromImage(image)
get_segmented_lungs(im_array[100], plot=True)