import torch
import torch.nn as nn
import torch.nn.functional as F

# from test import Judge

class NomalBlock(nn.Module):
    
    '''
    实现子module: Residual Block
    '''
    
    def __init__(self,inchannel,outchannel,stride=1):
        
        super(NomalBlock,self).__init__()
        
        
#         self.nomal = nn.ModuleList([nn.Sequential(nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
#                                                   nn.BatchNorm2d(outchannel),
#                                                   nn.ReLU(inplace=True)),
#                                     nn.Sequential(nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
#                                                   nn.BatchNorm2d(outchannel))])
        self.nomal = nn.Sequential(nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
                                                  nn.BatchNorm2d(outchannel),
#                                                   nn.ReLU(inplace=True)
                                                  nn.Mish()
                                  )


    def forward(self,x):

#         out = x
#         for model in self.nomal:
        x = self.nomal(x)
        return x

class PreInputBlock(nn.Module):

    '''
    实现子module: Residual Block
    '''

    def __init__(self,outchannel,stride=1):

        super(PreInputBlock,self).__init__()

        self.pre = nn.ModuleList([nn.Sequential(nn.Conv2d(3, outchannel,1,stride, bias=False),
                                                nn.BatchNorm2d(outchannel))])


    def forward(self,x):
        out = x
        for model in self.pre:
            out=model(out)
        return out

class Skipnet18(nn.Module):

    '''
    实现主module：SkipNet34
    SkipNet34 包含多个正常layer和多个数据输入预处理layer，
    用子module来实现nomal block 和pre-input block，用_make_pre_layer和_make_layer函数来实现layer
    '''
    def __init__(self,num_classes=12):

        super(Skipnet18,self).__init__()
        # k -- 第k层的input
        self.k = 1
        # 前几层图像转换
        self.first=nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            nn.Mish(),
            # nn.MaxPool2d(3,1,1)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        # 分别有6，8，12，6个数据预输入层  ，5+4+4+4 = 17层input
        self.total_pre_layers = nn.ModuleList()
        self.total_pre_layers.extend(self._make_pre_layer(outchannel=64,bloch_num=5,stride=1)) # 64
        self.total_pre_layers.extend(self._make_pre_layer(outchannel=128,bloch_num=4,stride=2))# 128
        self.total_pre_layers.extend(self._make_pre_layer(outchannel=256,bloch_num=4,stride=4))# 256
        self.total_pre_layers.extend(self._make_pre_layer(outchannel=512,bloch_num=6,stride=8))# 512

        #64*16*16
        # 重复的layer，分别有3，4，6，3个nomal block  ，残差结构，因为NomalBlock内只有一个conv，所以是4,4,4,4,
        self.total_layers = nn.ModuleList()
        self.total_layers.extend(self._make_layer(64,64,4))
        self.total_layers.extend(self._make_layer(64,128,4,stride=2))
        self.total_layers.extend(self._make_layer(128, 256, 4, stride=2))
        self.total_layers.extend(self._make_layer(256,512,6,stride=2))
        self.fc=nn.Linear(512*16,num_classes)
    def _make_pre_layer(self,outchannel,bloch_num,stride=1):
        '''
        构建数据输入层
        '''
        layers = nn.ModuleList()
        for i in range(bloch_num):
            layers.append(PreInputBlock(outchannel,stride))
        return layers



    def _make_layer(self,inchannel,outchannel,bloch_num,stride=1):

        '''
        构建layer,包含多个nomal block
        '''
        layers=nn.ModuleList()
        layers.append(NomalBlock(inchannel,outchannel,stride)) # 第一层的残差结构
        for i in range(1,bloch_num):
            layers.append(NomalBlock(outchannel,outchannel,1)) # 第一层后面的残差结构


        return layers

    def forward(self,x):
#         print(“网络”,self.k)
        # 如果从第一层开始输入，不需要pre_layers
        if self.k == 1:
#             x = self.layer1(x)
            x = self.first(x)
            for model in self.total_layers:
                    x = model(x)
        # 如果不是从第一层开始输入，则需要从第k层input开始，计算后面的残差结构和当前input
        else:
            x = self.total_pre_layers[self.k-2](x)
            if self.k < 20:
                for i,layers in enumerate(self.total_layers): # enumerate--遍历layer，i是每个layer的索引
                    if i >= self.k-2:
                            x = layers(x)

        x=F.avg_pool2d(x,7,stride=3)
        x=x.view(x.size(0),-1)
        return self.fc(x)

    
    def GetK(self,k):
        
        self.k = k
    

if __name__ == '__main__':

    # print(Skipnet18())
    
#     print()
    model = Skipnet18()
    # print(len(model.total_pre_layers))
    # print(len(model.total_layers))
#     print(model.total_layers[1])

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    