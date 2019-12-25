import math
import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot

SHAPE_SIDE = 0  # the size of one side of the square picture


#  sum of squares
def sumofsq(mas):
    sum = 0
    for el in mas:
        sum = sum + el ** 2
    return math.sqrt(sum)


#  sigmoid activation function
def activfunc(x):
    return 1 / (1 + 2.718 ** (-x))


#  derivative of sigmoid activation function
def derivativeofactivfunc(x):
    return activfunc(x) * (1 - activfunc(x))


#  creating an matrix with zero values
def getMatrix(a, b):
    return np.array([np.array([0.0 for j in range(b)]) for i in range(a)])


#  creating an array with zero values
def getMas(a):
    return np.array([0.0 for i in range(a)])


class PersiptronNetwork:
    def __init__(self, lengthSide, numScr, numClass=5):
        self.S = getMatrix(lengthSide ** 2, numScr)  # начало матрицы веса
        self.E = getMatrix(numScr, numClass)  # конец весовой матрицы
        self.lengthSide = lengthSide  # размер изображения
        self.numClass = numClass  # колличество классов
        self.Q = self.X = self.G = self.sum = getMas(
            numScr)  # входной порог, входные значения, значения скрытого слоя, взвешенные суммы 2 слоя
        self.T = self.Y = self.sum2 = getMas(numClass)  # выходной порог, выходные значения, взвешенные суммы 2 слоя
        self.numScr = numScr
        self.a = 0.1  # скорость обучения
        self.b = 0.1

    def training(self, masIn, masOut, n=5):
        while True:
            sum = 0
            for i in range(n):
                sum += sumofsq(self.teach(masIn[i], masOut[i]))
            if sum < 0.1:
                break
            print(sum)

    def teach(self, In, Out):

        self.work(In)
        Err = []  # расчет ошибки выходного слоя
        for i in range(self.numClass):
            Err.append(Out[i] - self.Y[i])

        for i in range(self.numClass):
            for j in range(self.numScr):
                self.E[j][i] = self.E[j][i] + self.a * Err[i] * self.G[j] * derivativeofactivfunc(self.sum2[i])
        for i in range(self.numClass):
            self.T[i] = self.T[i] + self.a * Err[i] * derivativeofactivfunc(self.sum2[i])

        Err2 = []  # расчет ошибки входного слоя
        for i in range(self.numScr):
            sum = 0
            for j in range(self.numClass):
                sum = sum + Err[j] * derivativeofactivfunc(self.sum2[j]) * self.E[i][j]
            Err2.append(sum)

        for i in range(self.numScr):
            for j in range(self.lengthSide ** 2):
                self.S[j][i] = self.S[j][i] + self.b * Err2[i] * self.X[j] * self.G[i] * (1 - self.G[i])
        for i in range(self.numScr):
            self.Q[i] = self.Q[i] + self.b * Err2[i] * self.G[i] * (1 - self.G[i])
        return Err

    def work(self, In):
        self.X = In

        for i in range(self.numScr):
            sum = 0
            for j in range(self.lengthSide ** 2):
                sum = sum + self.S[j][i] * self.X[j]
            self.sum[i] = sum
            self.G[i] = 1 / (1 + 2.718 ** (-(sum + self.Q[i])))

        for i in range(self.numClass):
            sum = 0
            for j in range(self.numScr):
                sum = sum + self.E[j][i] * self.G[j]
            self.sum2[i] = sum
            self.Y[i] = activfunc(sum + self.T[i])

        return self.Y

    def startInit(self):
        for i in range(self.lengthSide ** 2):
            for j in range(self.numScr):
                self.S[i][j] = random.uniform(-1, 1)

        for i in range(self.numScr):
            for j in range(self.numClass):
                self.E[i][j] = random.uniform(-1, 1)

        for i in range(self.numScr):
            self.Q[i] = random.uniform(-1, 1)

        for i in range(self.numClass):
            self.T[i] = random.uniform(-1, 1)
    def saveWeights(self,path = "weigths"):
        np.save(path + "S.npy",self.S)
        np.save(path + "E.npy",self.E)
        np.save(path + "Q.npy",self.Q)
        np.save(path + "T.npy",self.T)
    def loadWeights(self,path = "weigths"):
        self.S = np.load(path + "S.npy")
        self.E = np.load(path + "E.npy")
        self.Q = np.load(path + "Q.npy")
        self.T = np.load(path + "T.npy")



#  open file and make mode '1'
def openfile():
    listimage = []
    for filename in os.listdir('teach'):
        image = Image.open(str('teach/' + filename))
        listimage.append(image.convert('1'))
    return listimage


#  binarization image in 1 and -1 format
def binimage(listimage):
    global SHAPE_SIDE
    SHAPE_SIDE = listimage[0].size[0]
    shapes = []
    for image in listimage:
        file = []
        for y in range(image.size[0]):
            for x in range(image.size[1]):
                if image.load()[x, y] == 0:
                    file.append(-1)
                else:
                    file.append(1)
        shapes.append(file)
    return shapes


#  adding noise to the picture
def pushNoise(shape, noise):
    r = SHAPE_SIDE ** 2 * (noise / 100)
    a = []
    for i in range(int(r)):
        a.append(random.randint(0, SHAPE_SIDE ** 2 - 1))
    for i in a:
        shape[i] = -shape[i]
    return shape



def main(mode = True):
    listimage = openfile()
    shapes = binimage(listimage)
    persiptron = PersiptronNetwork(SHAPE_SIDE, 16)
    persiptron.startInit()
    
    masOut = np.array([np.array([1 if i == j else 0 for j in range(len(listimage))]) for i in range(len(listimage))])
    if mode == True:
        persiptron.training(shapes, masOut)
        persiptron.saveWeights()
    else:
        persiptron.loadWeights()
    noise = 5
    picture = 1
    result = {a: 0 for a in range(noise, 100, noise)}
    while noise < 100:
        for shape in shapes:
            shape = pushNoise(shape, noise)
            answer = persiptron.work(shape)
            print("Picture {}.txt - noize {} - answer {}".format(picture, noise, answer.argmax() + 1))
            if picture == answer.argmax() + 1:
                result[noise] += 100 / len(listimage)
            picture += 1
        noise += 5
        picture = 1
    print(result)
    pyplot.plot(list(result.keys()), list(result.values()))
    pyplot.show()

if __name__ == '__main__':
    main(False)
