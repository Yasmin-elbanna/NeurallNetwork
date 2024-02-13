
import cv2
import numpy as np


P = []
T = []
weight1 = np.array([])
weight2 = np.array([])
image = None
text = "Type is : "


def start():
    global weight1,weight2, T, P
    for i in range(10):
        P.append( get_feature(cv2.imread(f"data2/saturn.{i}.jpg",cv2.IMREAD_GRAYSCALE)) )
        T.append(1)
        P.append( get_feature(cv2.imread(f"data2/earth.{i}.jpg",cv2.IMREAD_GRAYSCALE)) )
        T.append(-1)
    P = np.array(P)
    T = np.array(T)
    weight1 = P
    weight2 =T
    print(weight1)
    print(weight2)






def get_feature(image):
    new = conv_relu(image)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = flatten(new)

    return new








def get_image():
    if(image == None):
        return f"data2/earth.0.jpg"
    
    return image.__dict__['url']








def conv_relu(image):
    mask = [[-1,-1,1],[0,1,-1],[0,1,1]]
    size1 = len(image) - 2
    size2 = len(image[0]) - 2
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(size1):
        for j in range(size2):
            x = 0
            for k in range(3):
                x += (image[i+k][j+0]*mask[k][0] + image[i+k][j+1]*mask[k][1] + image[i+k][j+2]*mask[k][2])
            new_image[i][j] = x if x > 0 else 0

    return new_image

def pooling(image):
    size1 = int(len(image)/2)
    size2 = int(len(image[0])/2)
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(0,size1):
        for j in range(0,size2):
            x = 0
            for k in range(2):
                x += (image[(i*2)+k][(j*2)+0] + image[(i*2)+k][(j*2)+1])/4
            new_image[i][j] = int(x)
    
    return new_image

def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(el)
    return new_image

def distance(a,b):
    sum =0
    for i in range(len(a)):
        sum+= pow(a[i]-b[i] ,2)  
    return np.sqrt(sum)  

def compititive():
    global image, weight1,weight2, text,T
    p = np.array(get_feature(cv2.imread(f'data2/' + image.__dict__['name'],cv2.IMREAD_GRAYSCALE)))
    a =[]
    print(p)
    for x in weight1:
        a.append(distance(x,p))
    winner = a.index(min(a))
    text = "Type is : Saturn" if T[winner]==1 else "Type is : Earth"


