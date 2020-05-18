import numpy as np
import cv2
import csv
import os
import random
import SimpleITK as sitk

rawPath='../data/LUNA16/raw/'
csvPath='./data/candidates.csv'
savePath='./system2/test_images/'
saveCSV='./system2/test_images_tag.csv'

test_tag = []

def get_data(mhd_dir):
    image=sitk.ReadImage(mhd_dir)
    im_array=sitk.GetArrayFromImage(image)
    return im_array

def normalize_hu(im):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image=im  #不改变原始数组
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image*255
    return image

#将结节坐标转为图像坐标
def word_to_image_coordinates(filename,nodules):
    #nodules是一个4维数组分别代表世界坐标中结节的x,y,z
    #函数返回图像坐标中结节的x,y,z
    filepath=rawPath+filename+'.mhd'
    itkimage = sitk.ReadImage(filepath)#读取.mhd文件
    Origin=itkimage.GetOrigin()
    SP=itkimage.GetSpacing()
    x, y, z = int((nodules[0]-Origin[0])/SP[0]), int((nodules[1]-Origin[1])/SP[1]), int((nodules[2]-Origin[2])/SP[2])
    return x,y,z



def make_savepath():
    if not os.path.exists(savePath):
        os.makedirs(savePath)


def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = open(file_name,'w',newline='')
    writer = csv.writer(file_csv)
    writer.writerow(["seriesuid","coordX","coordY","class"])
    for data in datas:
        writer.writerow(data)

def select_data():
    #随机取10%
    if random.random() < 0.1:
        return True
    else:
        return False

def search_tag_savedata():
    csvdata = csv.reader(open(csvPath, 'r'))
    idx=0    #记录图像保存进度
    idxcsv=0    #记录csv读取进度
    for raw in csvdata:
        idxcsv+=1
        if idxcsv%1000 == 0:
            print('----------csv:'+str(idxcsv)+'-----------')
        
        mhdPath = rawPath+raw[0]+'.mhd'
        if not os.path.exists(mhdPath):
            continue

        if raw[4] == str(0):    #降低假结节数目比例
            if not select_data():  
                continue
        
        '''if(raw[0] == 'seriesuid'):  #跳过表题
            continue'''
        
        x,y,z = word_to_image_coordinates(raw[0],[float(raw[1]),float(raw[2]),float(raw[3])])
        
        image = get_data(mhdPath)[z]
        image = normalize_hu(image)
        t = ['image_'+str(idx)+'.jpg',x,y,raw[4]]
        cv2.imwrite(savePath+'image_'+str(idx)+'.jpg',image)
        test_tag.append(t)

        idx+=1
        if idx%10 == 0:
            print('------saveim:'+str(idx)+'-------')
        if idx == 500:
            break
    
    data_write_csv(saveCSV,test_tag)
    print('-----------finished-------------')


if __name__ == '__main__':
    make_savepath()
    search_tag_savedata()
