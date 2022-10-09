import os
import numpy as np
import torch,torchvision
import cv2 as cv
from tqdm import tqdm
 
def NormalizeImNet(Image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for channel, _ in enumerate(Image):
        Image[channel] -= mean[channel]
        Image[channel] /= std[channel]
    return Image

def UnNormalizeImnet(Image):
    reverseMean = [-0.485, -0.456, -0.406]
    reverseStd = [1/0.229, 1/0.224, 1/0.225]
    for channel,_ in enumerate(Image):
            Image[channel] /= reverseStd[channel]
            Image[channel] -= reverseMean[channel]
    return Image

def arrayToTensor(Image,blur=False,normalize=False):
    if blur:
        Image = cv.GaussianBlur(Image,(3,3),0)

    if not torch.is_tensor(Image):
        try:
            Image = torch.Tensor(Image)
        except Exception as e:
            print("Could not transfrom data to Tensor type")
    #If the batch size is included remove it.
    if Image.dim() ==4:
        Image = Image[0]
    if normalize:
        Image =NormalizeImNet(Image)
    Image.unsqueeze_(0)
    Image.requires_grad_(True)
    return Image


def tensorToArray(Tensor,unNormalize=False): 
    Image = np.float32(Tensor[0].clone().data.numpy())
    if unNormalize:
        Image= UnNormalizeImnet(Image)
    Image[Image > 1] = 1
    Image[Image < 0] = 0
    return Image

def storeImage(Image:np.float32,path:str):
    Image = np.float32(Image.transpose(1,2,0))
    Image = np.uint8(Image* 255)
    cv.imwrite(path,cv.cvtColor(Image,cv.COLOR_BGR2RGB))

class ClassModelGen():
    def __init__(self,model,target,clipping=False) -> None:
        self.model = model
        self.target = target
        self.processImage= None
        self.clipping =clipping
        self.model.eval()
        self.generatedImage = np.float32(np.random.uniform(low=0,high=1,size=(3,224,224)))
        if not os.path.exists('./generatedImage'):
            os.mkdir('./generatedImage')
        self.path = f'./generatedImage/class{self.target}A.png'
        

    def generate(self,iterations=500):
        learningRate = 6
        for i in tqdm(range(1,iterations)):
            if (i+1)%6==0:
                blur =True
            else:
                blur=False
            self.processImage = arrayToTensor(self.generatedImage,blur=blur,normalize=True)
            optimizer =torch.optim.SGD([self.processImage],lr=learningRate,weight_decay=0.005)
            output = self.model(self.processImage)
            classLoss = -output[0,self.target]
            classLoss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.05)
            optimizer.step()
            self.model.zero_grad()
            self.generatedImage =tensorToArray(self.processImage,unNormalize=True)        
            if (i+1) % 10 == 0 or (i+1) == iterations:
                storeImage(self.generatedImage,self.path)




if __name__ == '__main__':
    # target_class = 319 # dragonfly
    target_class = 71 # scorpion

    # pretrained_model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    pretrained_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

    csig = ClassModelGen(pretrained_model, target_class,clipping=True)
    csig.generate()   
