import numpy as np
import SimpleITK as sitk

rawPath='../data/LUNA16/seg-lungs-LUNA16/'

def get_data(mhd_dir):
    image=sitk.ReadImage(mhd_dir)
    im_array=sitk.GetArrayFromImage(image)
    print(np.max(im_array),np.min(im_array))
    return im_array

im = get_data(rawPath+'1.3.6.1.4.1.14519.5.2.1.6279.6001.100530488926682752765845212286.mhd')
print(im)