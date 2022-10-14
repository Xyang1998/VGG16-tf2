import sys
import os

import PIL.Image
import cv2
import numpy
import numpy as np
from PIL import Image

trainpathlist=['1.2.3','1.2.3.4','4','5.6.7.8','5.8','6.7']
trainnum=[1367,1553,1323,1425,1479,1370]
trainlist=['123','1234','4','5678','58','67']
batchsize=500
typedict={'123':0,'1234':1,'4':2,'5678':3,'58':4,'67':5}
def gettype(a):#123=0,1234=1,4=2,5678=3,58=4,67=5
    b=typedict[a]
    return b
def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return image_lap
def loadseq():
    loadlist=[]
    if not os.path.exists('./train.txt'):
        f = open('./train.txt', 'w')
        for i in range(0, 6):
            namelist = os.listdir('./Train set/' + trainpathlist[i])
            for name in namelist:
                if ('.jpg' in name):
                    f.write('./Train set/' + trainpathlist[i] + '/' + name + ':' + trainlist[i] + '\n')
        f.close()
    for a in range(8517):
      loadlist.append(a)

    list=np.random.permutation(loadlist)
    return list

def get_batchlist(list):
    x_trainlist=[]
    y_trainlist=[]
    f=open('./train.txt','r')
    contents=f.readlines()
    for a in list:
        i=contents[a].split(':')[0]
        j=gettype(contents[a].split(':')[1].split('\n')[0])
        x_trainlist.append(i)
        y_trainlist.append(j)
    return x_trainlist,y_trainlist

def get_batch(x_trainlist,y_trainlist):
    y_train=y_trainlist
    x_train=[]
    for x in x_trainlist:
     img=Image.open(x)
     #img.show()
     img=np.array(img)
     img=laplacian(img)
     a=PIL.Image.fromarray(img)
     #a.show()
     x_train.append(img)
    x_train=np.array(x_train)
    x_train=x_train.astype(float)
    y_train=np.array(y_train,dtype=float)
    return x_train,y_train
def load_testseq():
    loadlist = []
    if not os.path.exists('./test.txt'):
      f=open('./test.txt','w')
      for i in range(0, 6):
          namelist=os.listdir('./Test set/'+trainlist[i])
          for name in namelist:
              if('.jpg' in name):
                  f.write('./Test set/'+trainlist[i]+'/'+name+':'+trainlist[i]+'\n')
      f.close()
    for a in range(2676):
      loadlist.append(a)

    list=np.random.permutation(loadlist)
    print(len(list))
    return list

def get_testbatchlist(list):
    x_testlist=[]
    y_testlist=[]
    f=open('./test.txt','r')
    contents=f.readlines()
    for a in list:
        print(a)
        i=contents[a].split(':')[0]
        j=gettype(contents[a].split(':')[1].split('\n')[0])
        x_testlist.append(i)
        y_testlist.append(j)
    return x_testlist,y_testlist

def get_testbatch(x_testlist,y_testlist):
    y_train=y_testlist
    x_train=[]
    for x in x_testlist:
     img=Image.open(x)
     img = np.array(img)
     img = laplacian(img)
     x_train.append(img)
    x_train=np.array(x_train)
    x_train=x_train.astype(float)
    y_train=np.array(y_train,dtype=float)
    return x_train,y_train

