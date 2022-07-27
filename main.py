import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
from sklearn.decomposition import PCA

def load_images():
    folders_chosen =[]
    allFiles = []
    allCounts = []
    allPaths = []
    for i in range(10):
        num = random.randint(0,3999)
        folders_chosen.append(num)
    for i in range(len(folders_chosen)):
        folder = folders_chosen[i]
        path ='train_data\\'+str(folder)
        allPaths.append(path)
        list = []
        count=0
        for file in os.listdir(path):
            count+=1
            list.append(file)

        allFiles.append(list)
        allCounts.append(count)
    for i in range(len(allFiles)):
        sampleSize = allCounts[i]
        if sampleSize>50:
            sampleSize = 50
        allFiles[i] = random.sample(allFiles[i],sampleSize)
    return allFiles,allPaths

def Unicolor():
    images = load_images()
    width = 64
    height = 64
    allDirectories = []
    Imagour = []
    ImagesInDirectory = []

    for w in images[1]:
        for file in os.listdir(w):
            path = w+'\\'+str(file)
            img = Image.open(path)
            cols = []
            for i in range(width):
                row= []
                for j in range(height):
                    colors = img.getpixel((i,j))
                    unicolor = 0.299 *colors[0] + 0.587 * colors[1] + 0.114 * colors[2]
                    row.append(unicolor)
                cols.append(row)
            Imagour.append(cols)
        ImagesInDirectory.append(Imagour)
    allDirectories.append(ImagesInDirectory)

def pcaFunc(pcaComponents):
    loadedImages = load_images()
    count = 0

    for i in range(10):
        folder =  loadedImages[0][i]
        for file in folder:
                count+=1
                path = loadedImages[1][i] + '\\'
                path = path + file
                print(str(count) + path)
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                #img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                plt.imshow(img)
                plt.show()
                print(img.shape)
                blue,green,red = cv2.split(img)


                '''fig = plt.figure(figsize=(15, 7.2))
                fig.add_subplot(131)
                plt.title("Blue Channel")
                plt.imshow(blue)
                fig.add_subplot(132)
                plt.title("Green Channel")
                plt.imshow(green)
                fig.add_subplot(133)
                plt.title("Red Channel")
                plt.imshow(red)
                plt.show()'''


                df_blue = blue/255
                df_green = green/255
                df_red = red/255

                pca_b = PCA(n_components=pcaComponents)
                pca_b.fit(df_blue)
                trans_pca_b = pca_b.transform(df_blue)
                pca_g = PCA(n_components=pcaComponents)
                pca_g.fit(df_green)
                trans_pca_g = pca_g.transform(df_green)
                pca_r = PCA(n_components=pcaComponents)
                pca_r.fit(df_red)
                trans_pca_r = pca_r.transform(df_red)
                #img_compressed = (np.dstack((red_inverted,green_inverted,blue_inverted))).astype(np.uint8)
                #plt.imshow(img_compressed)
                #plt.show()
                b_arr = pca_b.inverse_transform(trans_pca_b)
                g_arr = pca_g.inverse_transform(trans_pca_g)
                r_arr = pca_r.inverse_transform(trans_pca_r)


                img_reduced = (cv2.merge((b_arr, g_arr, r_arr)))


                fig = plt.figure(figsize=(10, 7.2))
                fig.add_subplot(121)
                plt.title("Original Image")
                plt.imshow(img)
                fig.add_subplot(122)
                plt.title("Reduced Image")
                plt.imshow(img_reduced)
                plt.show()
pcaFunc(50)