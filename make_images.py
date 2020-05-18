from scipy import ndimage as ndi
import SimpleITK as sitk
import numpy as np
import cv2
import csv
import os
import random

rawPath='./data/raw/'
pngPath='./data/segmenteddata/'
csvPath='./data/candidates.csv'
savePath='./data/finalData/'
tagSavePath='./data/tag.csv'


Tag=[] #[filename,x,y,z,c] 写入csv

def make_savepath():
    if not os.path.exists(savePath):
        os.makedirs(savePath)


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

def get_data(filename,x,y,z,cl):
    impath=pngPath+filename+'/'+str(z)+'.png'
    image=cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
    tag=[filename,x,y,z,cl]
    return image,tag

def select_data():
    #随机取10%
    if random.random() < 0.1:
        return True
    else:
        return False

def get_subimage(image, x,y,width=50):
        """
        Returns cropped image of requested dimensiona
        """
        subImage = image[int(y-width/2):int(y+width/2),\
         int(x-width/2):int(x+width/2)]
        return subImage



def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = open(file_name,'w',newline='')
    writer = csv.writer(file_csv)
    writer.writerow(["seriesuid","coordX","coordY","coordZ","class"])
    for data in datas:
        writer.writerow(data)

def search_candidates_savedata():
    csvdata = csv.reader(open(csvPath, 'r'))
    idx=0    #记录子图像保存进度
    idxcsv=0    #记录csv读取进度
    for raw in csvdata:
        idxcsv+=1
        if idxcsv%1000 == 0:
            print('----------csv:'+str(idxcsv)+'-----------')

        if raw[4] == str(0):    #降低假结节数目比例
            if not select_data():  
                continue

        if(raw[0] == 'seriesuid'):
            continue
        
        mhdfile=rawPath+raw[0]+'.mhd'
        if not os.path.exists(mhdfile):
            continue

        nodules=[float(raw[1]),float(raw[2]),float(raw[3])]
        x,y,z=word_to_image_coordinates(raw[0],nodules)
        image,tag=get_data(raw[0],x,y,z,raw[4])    #tag:[filename,x,y,z,c]

        subimage=get_subimage(image,tag[1],tag[2])

        cv2.imwrite(savePath+'image_'+str(idx)+'.jpg',subimage)
        Tag.append(tag)

        idx+=1
        if idx%100 == 0:
            print('------subim:'+str(idx)+'-------')
    
    print('------image save over-------')

    #write Tag
    data_write_csv(tagSavePath,Tag)
    print('-------tag write over------')


if __name__ == '__main__':
    make_savepath()
    search_candidates_savedata()
