import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./archive/train/angry/Training_3908.jpg')
img = cv2.resize(img,(48,48))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class Conv2d: 
    def __init__(self,input,kernelSize):
        self.height , self.width = input.shape
        self.kernelSize = kernelSize
        self.kernel =np.random.rand(kernelSize,kernelSize)
        self.img_2d = np.zeros((self.height,self.width))
        self.input = input
        self.padding = int(kernelSize/2)
    def padd(self):
        self.input = np.pad(self.input,((self.padding,self.padding),(self.padding,self.padding)),mode = 'constant', constant_values=0)
        self.height = self.height+self.kernelSize-1
        self.width = self.width+ self.kernelSize-1
    def operator(self):
        self.padd()
        for row in range(0,self.height-self.kernelSize+1):
            for col in range(0,self.width-self.kernelSize+1):
                sum = 0
                for i in range(0,self.kernelSize):
                    for j in range(0,self.kernelSize):
                        sum+= self.input[row+i][col+j]*self.kernel[i][j]
                self.img_2d[row][col] = sum
        self.height = self.height - self.kernelSize +1
        self.width = self.width - self.kernelSize +1
        return self.img_2d

class Pooling:
    def __init__(self, input, max_or_avarage=0):
        self.height, self.width = input.shape
        self.height_result = int(self.height/2)
        self.width_result = int(self.width/2)
        print(self.height_result,self.width_result)
        self.result = np.zeros((self.height_result,self.width_result))
        self.input = input
        self.max_or_avarage = max_or_avarage
    def operator(self):
        if(self.max_or_avarage==0):
            for row in range (0,self.height,2):
                for col in range (0,self.width,2):
                    self.result[row//2][col//2] = max(self.input[row][col],self.input[row+1][col],self.input[row][col+1],self.input[row+1][col+1])
            return self.result
        if(self.max_or_avarage==1):
            for row in range (0,self.height,2):
                for col in range (0,self.width,2):
                    self.result[row//2][col//2] = (self.input[row][col]+self.input[row+1][col]+self.input[row][col+1]+self.input[row+1][col+1])/4
            return self.result
        
img_covolution = Conv2d(img, 3).operator()
img_pooling = Pooling(img_covolution,0).operator()
print(img_pooling.shape)
plt.imshow(img_pooling, cmap='gray')
plt.show()