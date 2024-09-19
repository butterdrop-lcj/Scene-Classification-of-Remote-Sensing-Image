# 使用cifar10数据集，跳跃网络
import os
from PIL import Image
import torch
from torchvision import transforms
from numpy import log2
import torch
import torch.nn.functional as F
from torch.nn.modules import flatten
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.pooling import MaxPool2d
import torchvision
from torch.utils.data import Dataset,DataLoader, dataloader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from model import Skipnet18
from torch.utils.data import Subset, DataLoader
import random
class RSSCN7Dataset(Dataset):
    def __init__(self, root_dir, train=True ,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        if self.train:
            self.data_path = os.path.join(self.root_dir, 'train')
        else:
            self.data_path = os.path.join(self.root_dir, 'test')

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    small_dataset_size = 240  # 假设构建一个包含100个样本的小数据集

    # from resnet_skip import ResNet_Skip

    # torch.manual_seed(1)
    transform_train = transforms.Compose([
        transforms.RandomCrop(128, padding=4),  # 先四周填充0，再把图像随机裁剪成128x128
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
       # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(128, padding=4),  # 先四周填充0，再把图像随机裁剪成128x128
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = RSSCN7Dataset(root_dir="./SIRI-WHU/train",train=True,transform=transform_train)
    test_dataset = RSSCN7Dataset(root_dir="./SIRI-WHU/test",train=False,transform=transform_test)

    # train_dataset = torchvision.datasets.MNIST(root="/home/featurize/work/data",train=True,transform=transform_train,download=True)
    # test_dataset = torchvision.datasets.MNIST(root="/home/featurize/work/data",train=False,transform=transform_test,download=True)


    # 确定大数据集的总样本数
    num1_samples = len(train_dataset)
    num2_samples = len(test_dataset)

    # 随机选择样本索引来构建小数据集
    random1_indices = random.sample(range(num1_samples), small_dataset_size)
    random2_indices = random.sample(range(num2_samples), small_dataset_size)

    # 使用随机索引创建小数据集
    small1_dataset = Subset(train_dataset, random1_indices)
    small2_dataset = Subset(train_dataset, random2_indices)

    #数据集长度
    train_data_size = len(small1_dataset)
    test_data_size = len(small2_dataset)
    print("训练集长度：{}".format(train_data_size))
    print("测试集长度：{}".format(test_data_size))

    train_dataloader = DataLoader(small1_dataset,batch_size=16,shuffle=False,num_workers=2)
    test_dataloader = DataLoader(small2_dataset,batch_size=16,shuffle=False,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #创建网络模型
    # test_net = skip_net_mnist()
    # test_net = SkipNetC2()
    test_net = Skipnet18()
    # if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   test_net = nn.DataParallel(test_net)

    # ../input/skipnetmodel/skip_net_CIFAR10_164 (3)
    # test_net.load_state_dict(torch.load("(1)—3k_nopre_mish_Skipnet18_0_50_128_0.01.pkl"))

    test_net.to(device)
    #创建损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    #创建优化器
    # learning_rata = 0.01#或者使用1e-2代替0.01
    learning_rate = 0.0002
    # optimizer = torch.optim.SGD(test_net.parameters(),lr=learning_rata,momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(test_net.parameters(), lr=learning_rate)


    def GetParametaer(net):
        # Get parameters of the net
        list = []
        for __, parameter in net.named_parameters():
            list.append(parameter)

        return list

    def GetNum(NumOfLayers,x):
        test_net.GetK(NumOfLayers)
        x = test_net(x)
        return x

    #bitch_size
    #可以得到每一层的输出

    def Get_layers_OutNums(x):
        list_3 = []
        for i in range(1,21): # 每一层的input
            temp = x
            temp = GetNum(i,temp) # 计算每一层input的output
            temp = F.softmax(temp,dim=1)
            list_3.append(temp)
        return list_3



    def LossOfEveryLayers(x, y,LossFunc):
        #Get the loss ofc every linaer layer
        loss_of_layer = []
        for i in range(20):
            loss_of_layer.append(LossFunc(x[i],y))
        return loss_of_layer.index(min(loss_of_layer))+1




    def OutPut(layers_out):
        layers_out_ = torch.stack(layers_out,0).to(device)
        temp2 = torch.max(layers_out_,2)[0].argmax(0)
        temp3 = torch.arange(0,len(temp2))
        output_  = layers_out_[temp2,temp3].to(device)
        return output_


    def returnk(layers_out):
        layers_out_ = torch.stack(layers_out,0).to(device)
        temp = torch.max(layers_out_,2)[0].argmax(0)
        temp2 = temp.tolist()
        k = max(temp2,key=temp2.count)+1
        return k


    plt.figure()

    #训练的次数
    total_train_step = 0
    #测试次数
    total_test_step = 0
    total_train_accuracy = 0
    # Decey_x = 0.5
    total_train_accuracy_list = []
    total_test_accuracy_list = []
    change_k = 0


    for i in range(20):
        print("第{}轮训练开始".format(i+1))
        total_train_loss = 0
        total_accuracy1 = 0
        total_accuracy2 = 0
        total_accuracy3 = 0

        test_net.train()
        for data in train_dataloader:

            imgs,labels = data
            imgs = imgs.to(device)
            imgs = imgs.view(-1,3,128,128)
            # imgs = imgs.view(-1,3,32,32)、
            labels = labels.to(device)
            #layers_out = Get_layers_OutNums(imgs)

            if i<change_k:
                k = 1
                test_net.GetK(k)
                output = test_net(imgs)
                loss = loss_fn(output,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_train_step = total_train_step + 1
                if total_train_step%100 == 0:
                    print("训练次数:{}，loss: {}".format(total_train_step,loss.item()))
                accuracy1 = (output.argmax(1) == labels).sum()
                total_accuracy1 += accuracy1

            else:
                layers_out = Get_layers_OutNums(imgs)


                k_1 = LossOfEveryLayers(layers_out,labels,loss_fn)
        #         print("训练",k_1)
                test_net.GetK(k_1)
        #         print(test_net.k)
                outputs1 = test_net(imgs)
                loss1 = loss_fn(outputs1,labels)



                k3 = returnk(layers_out)
        #         print("训练",k3)
                test_net.GetK(k3)
                outputs3 = test_net(imgs)
        #         print(test_net.k)
                loss3 = loss_fn(outputs3,labels)

        #         loss = loss3

        #
                outputs2 = OutPut(layers_out)
                loss2 = loss_fn(outputs2,labels)
                loss = loss1+loss2+loss3 # 三种路径获取的损失的累加
        #         loss = loss2+loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_step = total_train_step + 1
                if total_train_step%100 == 0:
        #             print("训练次数:{}，loss3: {}".format(total_train_step,loss3.item()))
        #             print("训练次数:{}，loss2: {}，loss3: {}".format(total_train_step,loss2.item(),loss3.item()))
        #             print("训练次数:{}，loss1: {}，loss2: {}".format(total_train_step,loss1.item(),loss2.item()))
                    print("训练次数:{}，loss1: {}，loss2: {}，loss3: {}".format(total_train_step,loss1.item(),loss2.item(),loss3.item()))
                accuracy1 = (outputs1.argmax(1)==labels).sum()
                accuracy2 = (outputs2.argmax(1)==labels).sum()
                accuracy3 = (outputs3.argmax(1)==labels).sum()
                total_accuracy1 += accuracy1
                total_accuracy2 += accuracy2
                total_accuracy3 += accuracy3
        if i<change_k:
            train_accurary1 = total_accuracy1/train_data_size
            print(total_train_loss)
            print("在训练集上的正确率：{}".format(train_accurary1))
        else:
            train_accurary1 = total_accuracy1/train_data_size
            train_accurary2 = total_accuracy2/train_data_size
            train_accurary3 = total_accuracy3/train_data_size
            print(total_train_loss)
            print("在训练集上的正确率：{}".format(train_accurary1))
            print("在训练集上的正确率（不用k）：{}".format(train_accurary2))
            print("在训练集上的正确率（使用计数k）：{}".format(train_accurary3))
        total_train_accuracy_list.append(train_accurary1)
        total_accuracy = 0
        #测试步骤
        # test_net.eval()
        total_test_loss = 0
        total_test_step = 0

        with torch.no_grad():
            for data in test_dataloader:
                imgs,labels = data
                imgs = imgs.to(device)
                # imgs = imgs.view(-1,1,28,28)
                imgs = imgs.view(-1,3,128,128)
                labels = labels.to(device)
                if i<change_k:
                    k=1
                    test_net.GetK(k)
                    output_s = test_net(imgs)

                else:
                    layers_out2 = Get_layers_OutNums(imgs)
                    k = returnk(layers_out2)
                    test_net.GetK(k)
                    output_s = test_net(imgs)

                loss = loss_fn(output_s,labels)
                total_test_loss = total_test_loss + loss.item()
                total_test_step = total_test_step + 1
                if total_test_step%40 == 0:
                   print("训练次数{}，loss{}".format(total_test_step,loss.item()))
                accuracy = (output_s.argmax(1)==labels).sum()
                total_accuracy += accuracy
        test_accurary = total_accuracy/test_data_size
        print("在测试集上的正确率：{}".format(test_accurary))
        print(total_test_loss)

        total_test_step +=1
        total_test_accuracy_list.append(test_accurary)
    #     time_end=time.time()
    #     print('totally cost',time_end-time_start)




    for i in range(len(total_test_accuracy_list)):
        with open('siri_3k_nopre_mish_Skipnet18_0_70_128_0.01.txt', 'a') as f:
            f.write('%d %.3f %.3f\n' % (i+1,total_train_accuracy_list[i],total_test_accuracy_list[i]))
    #     with open('skip18_train_accura/te_records_k.txt', 'a') as f:
    #         f.write('%d %.3f\n' % (i+1,total_train_accuracy_list[i]))
    torch.save(test_net.state_dict(),"siri_3k_nopre_mish_Skipnet18_0_70_128_0.01.pkl")