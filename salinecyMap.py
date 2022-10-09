

import numpy as np
import torch,torchvision
import cv2 as cv


def NormalizeImNet(Image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    Image/=255
    for channel, _ in enumerate(Image):
        Image[channel] -= mean[channel]
        Image[channel] /= std[channel]
    return Image

def arrayToTensor(Image,normalize=False):

    Image = (cv.resize(Image,(224,224))).transpose(2,0,1)
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


def saveGrayImage(image,path):
    grayImage = np.sum(np.abs(image),axis=0)
    min = grayImage.min()
    max = grayImage.max()
    grayImage= np.clip((grayImage-min)/(max-min),0,1)
    grayImage = np.uint8(grayImage*255)
    cv.imwrite(path,grayImage)

    


class VanillaGrdientMap():

    def __init__(self,model) -> None:
        self.model = model
        self.gradient =None
        self.registerHook()

    def registerHook(self):
        def hookFn(module,gradIn,_):
            self.gradient = gradIn[0]

        firstLayer = list(self.model.features._modules.items())[0][1]
        firstLayer.register_full_backward_hook(hookFn)
    def generateMap(self,Image,target):
        output = self.model(Image)

        gradTarget = torch.zeros(output.shape,dtype=torch.float32)
        gradTarget[0,target] =1 # gradient w.r.t activated class

        output.backward(gradTarget)
        return self.gradient.data.numpy()[0] # [BxCxHxW]---> [CxHxW]


if __name__=="__main__":
    original = (cv.cvtColor(cv.imread('./gorillas.jpg'),cv.COLOR_BGR2RGB))

    imageTensor = arrayToTensor(original,normalize=True)

    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    salMap = VanillaGrdientMap(model)
    gradImage =salMap.generateMap(imageTensor,366) #366 is gorilla

    saveGrayImage(gradImage,'./grayIm.png')
