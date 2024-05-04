import torch 
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

imageSize = (256,256)

# file path
originFilePath = "./malimg_dataset/malimg_paper_dataset_imgs"
resizeFilePath = "./malimg_dataset/malimg_paper_dataset_imgs_resize"

def resizeAllImage(imagePath):
    im = Image.open(imagePath)
    # resize image
    resizeImage = im.resize(imageSize, Image.LANCZOS)
    imageArray = np.array(resizeImage)
    savePath = './malimg_dataset/malimg_paper_dataset_imgs_resize/' + imagePath.split('/')[-2] + '/' + imagePath.split('/')[-1]
    if not os.path.exists('./malimg_dataset/malimg_paper_dataset_imgs_resize/' + imagePath.split('/')[-2]):
        os.makedirs('./malimg_dataset/malimg_paper_dataset_imgs_resize/' + imagePath.split('/')[-2])
    resizeImage.save(savePath)
    
    return imageArray

def readData():
    img = []
    label = []

    # get all folder name
    imagePathList = []
    for folderName in os.listdir(resizeFilePath):
        imagePathList.append(resizeFilePath + '/' + folderName)

    # read all image
    for folder in tqdm(imagePathList):
        labelName = folder.split('/')[-1]
        for imageFile in tqdm(os.listdir(folder),leave=False):
            imageArray = np.array(Image.open(folder + '/' + imageFile))

            if len(imageArray.shape) == 2:
                imageArray = imageArray[:, :, np.newaxis]

            img.append(imageArray)
            label.append(labelName)

    img = np.stack(img, axis=0)
    return img, label, len(imagePathList)