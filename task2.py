
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

x = [[5.9, 3.2], [4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5 ,4.2], [5.0 ,3.0],[4.9 ,3.1],[6.7 ,3.1],[5.1 ,3.8],[6.0 ,3.0]]
u1 = [6.2, 3.2] #(red),
u2 = [6.6, 3.7] #(green)
u3 = [6.5, 3.0]  #(blue)

def calMin (a,b,c):
    a1=0
    b1=0
    c1=0
    if (a<b):
        if (a<c):
            #a smallest
            a1=1
        else:
            #c smallest
            c1=3
    elif (b<a):
        if(b<c):
            #b smallest
            b1=2
        else:
            #c smallest
            c1=3
    return a1,b1,c1;


def newCentre(c1,c2,c3):
    u1 = []
    u2=[]
    u3=[]
    u1 = np.mean(c1, axis =0)
    u2 = np.mean(c2, axis =0)
    u3 = np.mean(c3, axis =0)
    return u1,u2,u3;


def formCluster(img, mat, k):
    distVector = np.empty((img.shape[0],k))
    for i in range(k):
        dist = (img - mat[i]) ** 2
        distVector[:, i] = np.sum(dist, axis=1)
    classifyVector = distVector.argmin(axis=1)
    print("classifyVector")
    return classifyVector

def newMean(img, classifyVector, k):
    print('new mean',k)
    i = 0
    count = [0] * k
    sum = [[0] * k for _ in range(0,3)]
    while (i < len(img)):
        count[classifyVector[i]] = count[classifyVector[i]] + 1
        j=0
        while j < 3:
            sum[j][classifyVector[i]] = sum[j][classifyVector[i]] + img[i][j]
            j=j+1
        i = i + 1

    mat = np.zeros((k,3))
    for i in range(k):
        for j in range(0,3):
            if count[i] != 0:
                mat[i][j] = sum[j][i] / count[i]
            else:
                continue;
    return mat
def quantize(img, k):
    meanMat = []
    for i in range(k):
        meanMat.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    #img3 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img2 = img.reshape((-1, 3))
    for i in range(0,4):
        classifyVector = formCluster(img2, meanMat, k)
        meanMat = newMean(img2, classifyVector, k)


    final_img = meanMat[classifyVector]
    final_img = np.reshape(final_img, img.shape)
    cv2.imwrite('task2_baboon_{}.jpg'.format(k),final_img)
    print('image')

def classify(x, u1,u2,u3):

    #calclualte distance between given inital points u1,u2,u3
    for i in range(0, 10):
        d1 = np.sqrt((u1[0] - x[i][0]) ** 2 + (u1[1] - x[i][1]) ** 2)
        d2 = np.sqrt((u2[0] - x[i][0]) ** 2 + (u2[1] - x[i][1]) ** 2)
        d3 = np.sqrt((u3[0] - x[i][0]) ** 2 + (u3[1] - x[i][1]) ** 2)

        #Calculate minimum distance and find which u1, u2 or u3 is closest
        m, n, p = calMin(d1, d2, d3)
        print(m, n, p)
        if (m == 1):
            c1.append(x[i])
        elif n == 2:
            c2.append(x[i])
        elif p == 3:
            c3.append(x[i])




#classify(x,u10,u20,u30)
if __name__ == "__main__":
    for i in range (0,2):
        #calssify
        #plot
        #recompute center

        #declare clusters
        c1=[]
        c2=[]
        c3=[]
        classify(x, u1, u2, u3)             #changes elements of clusters c1,c2,c3

        for k in range(0,int (np.size(c1)/2)):
            plt.scatter(c1[k][0], c1[k][1], s=100, c=['#ff0000'], marker='^', edgecolors='#008888')
            plt.text(c1[k][0], c1[k][1], '({},{})'.format(c1[k][0], c1[k][1]))
        for t in range(0, int(np.size(c2) / 2)):
            plt.scatter(c2[t][0], c2[t][1], s=100, c=['#00ff00'], marker='^', edgecolors='#008888')
            plt.text(c2[t][0], c2[t][1], '({},{})'.format(c2[t][0], c2[t][1]))
        for k in range(0, int(np.size(c3) / 2)):
            plt.scatter(c3[k][0], c3[k][1], s=100, c=['#0000ff'], marker='^', edgecolors='#008888')
            plt.text(c3[k][0], c3[k][1], '({},{})'.format(c3[k][0], c3[k][1]))
        u1, u2, u3 = newCentre(c1, c2, c3)
        if i==0:
            plt.savefig('task2_iter1_a.jpg')
        else:
            plt.savefig('task2_iter2_a.jpg')
        plt.scatter(u1[0], u1[1], s=100, c='#ff0000')
        plt.text(u1[0], u1[1], '({},{})'.format(round(u1[0], 2), round(u1[1], 2)))
        plt.scatter(u2[0], u2[1], s=100, c='#00ff00')
        plt.text(u2[0], u2[1], '({},{})'.format(round(u2[0], 2), round(u2[1], 2)))
        plt.scatter(u3[0], u3[1], s=100, c='#0000ff')
        plt.text(u3[0], u3[1], '({},{})'.format(round(u3[0], 2), round(u3[1], 2)))

        if i==0:
            plt.savefig('task2_iter1_b.jpg')
        else:
            plt.savefig('task2_iter2_b.jpg')

        plt.clf()



    img = cv2.imread('baboon.png')
    quantize(img, k=3)
    quantize(img, k=5)
    quantize(img, k=10)
    quantize(img, k=20)



