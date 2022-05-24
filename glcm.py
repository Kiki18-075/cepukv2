import numpy as np
import cv2
import math

class GLCM:
    #maximum number of gray levels
    def __init__(self):
        self.gray_level = 256

    def maxGrayLevel(self, img):
        max_gray_level=0
        (height,width)=img.shape
        for y in range(height):
            for x in range(width):
                if img[y][x] > max_gray_level:
                    max_gray_level = img[y][x]
        return max_gray_level+1

    def getGlcm(self, input, degree, distance):
        srcdata=input.copy()
        ret=[[0.0 for i in range(self.gray_level)] for j in range(self.gray_level)]
        (height,width) = input.shape

        max_gray_level= self.maxGrayLevel(input)
        #If the number of gray levels is greater than gray_level, reduce the gray level of the image to gray_level and reduce the size of the gray level co-occurrence matrix
        if max_gray_level > self.gray_level:
            for j in range(height):
                for i in range(width):
                    srcdata[j][i] = srcdata[j][i]*self.gray_level / max_gray_level

        if degree == 0:
            for j in range(height - 0):
                for i in range(width - distance):
                    rows = srcdata[j][i]
                    cols = srcdata[j + 0][i+ distance]
                    ret[rows][cols]+=1.0
        elif degree == 45:
            for j in range(height - distance):
                for i in range(width - distance):
                    rows = srcdata[j][i]
                    cols = srcdata[j + distance][i+ distance]
                    ret[rows][cols]+=1.0
        elif degree == 90:
            for j in range(height - distance):
                for i in range(width):
                    rows = srcdata[j][i]
                    cols = srcdata[j - distance][i]
                    ret[rows][cols]+=1.0
        elif degree == 135:
            for j in range(distance, height):
                for i in range(distance, width):
                    rows = srcdata[j][i]
                    cols = srcdata[j - distance][i - distance]
                    ret[rows][cols]+=1.0

        for i in range(self.gray_level):
            for j in range(self.gray_level):
                ret[i][j]/=float(height*width)

        return ret

    def feature_computer(self,p):
        # Con=0.0 #contras
        Ent=0.0 #entropy
        Hom=0.0   #homogenity
        Diss=0.0 #dissimilarty
        Idm=0.0 #IDM
        Asm=0.0 #Angle second moment
        Eng=0.0 #energy
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                # Con+=p[i][j] * np.power(i - j, 2)
                Asm+=p[i][j]*p[i][j]
                Diss+=p[i][j]*np.abs(i-j)
                Eng+= np.power(p[i][j], 2)
                Hom+= p[i][j] / (1 + np.power(i - j, 2))
                Idm+=p[i][j]/(1+(i-j)*(i-j))
                if p[i][j]>0.0:
                    Ent+=p[i][j]*math.log(p[i][j])
        return Eng,Diss,Hom,Idm,Ent,Asm
    
    #get feature data training
    def get_feature(self, path):
        img =cv2.imread(path, 0)
        glcm = self.getGlcm(img, 0, 1)
        Eng,diss,hom,idm,ent,Asm= self.feature_computer(glcm)
        return Eng,diss,hom,idm,ent,Asm