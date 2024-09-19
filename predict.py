# 使用cifar10数据集，跳跃网络

from numpy import log2
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader, dataloader
import torch
import torch.nn.functional as F
from torch.nn.modules import flatten
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.pooling import MaxPool2d
import torchvision
from torch.utils.data import DataLoader, dataloader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from model import Skipnet18
from sklearn.metrics import classification_report
# import seaborn as sns
# from resnet_skip import ResNet_Skip

class RSSCN7Dataset(Dataset):
    def __init__(self, root_dir, train=True ,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        if self.train:
            self.data_path = os.path.join(self.root_dir, 'train')
        else:
            self.data_path = os.path.join(self.root_dir, 'test')

        for label, class_name in enumerate(self.classes):
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

    # torch.manual_seed(1)
    transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    #     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.RandomCrop(128, padding=4),  # 先四周填充0，再把图像随机裁剪成128x128
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
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
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_dataset = torchvision.datasets.CIFAR10(root="../input/cifar10",train=True,transform=transform_train,download=True)
    # test_dataset = torchvision.datasets.CIFAR10(root="../input/cifar10",train=False,transform=transform_test,download=True)

    train_dataset = RSSCN7Dataset(root_dir="./SIRI-WHU/train",train=True,transform=transform_train)
    test_dataset = RSSCN7Dataset(root_dir="./SIRI-WHU/test",train=False,transform=transform_test)


    #数据集长度
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("训练集长度：{}".format(train_data_size))
    print("测试集长度：{}".format(test_data_size))

    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=2)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=2)

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
    test_net.load_state_dict(torch.load("./siri_3k_nopre_mish_Skipnet18_0_70_128_0.01.pkl"))
    # test_net.load_state_dict(torch.load("./ca.pkl"))
    test_net.to(device)
    #创建损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    #创建优化器
    learning_rata = 0.01#或者使用1e-2代替0.01
    optimizer = torch.optim.SGD(test_net.parameters(),lr=learning_rata,momentum=0.9, weight_decay=5e-4)



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
        for i in range(1,21):
            temp = x
            temp = GetNum(i,temp)
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


    total_accuracy = 0
        #测试步骤
        # test_net.eval()
    total_test_loss = 0
    total_test_step = 0
    # 测试模型
    correct = [0] * 12
    total = [0] * 12
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            # images = images.view(-1,3,32,32)
            labels = labels.to(device)
            layers_out2 = Get_layers_OutNums(images)

            k = returnk(layers_out2)
            test_net.GetK(k)
            outputs = test_net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())
            for i in range(len(labels)):
                label = labels[i]
                total[label] += 1
                if predicted[i] == label:
                    correct[label] += 1

    # 输出每个类别的准确率
    for i in range(12):
        print('Accuracy of %5s : %.3f %%' % (
            test_dataset.classes[i], 100 * correct[i] / total[i]))


    # 将预测结果转换为标签
    y_pred_labels = np.array(y_pred)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_labels)

    print(cm)


    # 绘制混淆矩阵
    # plt.figure(figsize=(10, 8))
    # sns.set(font_scale=1.4)
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # 添加数值
    for i in range(12):
        for j in range(12):
            plt.annotate(str(cm[i][j]), xy=(j, i), horizontalalignment='center', verticalalignment='center')

    plt.colorbar()
    tick_marks = np.arange(12)
    # plt.xticks(tick_marks, ['Grass', 'Field', 'Industry', 'RiverLake', 'Forest', 'Resident', 'Parking'], rotation=45)
    # plt.yticks(tick_marks, ['Grass', 'Field', 'Industry', 'RiverLake', 'Forest', 'Resident', 'Parking'])
    plt.xticks(tick_marks, ['agriculture', 'commercial', 'harbor', 'idle_land', 'industrial', 'meadow', 'overpass', 'park', 'pond', 'residential', 'river', 'water'], rotation=45)
    plt.yticks(tick_marks, ['agriculture', 'commercial', 'harbor', 'idle_land', 'industrial', 'meadow', 'overpass', 'park', 'pond', 'residential', 'river', 'water'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # plt.show()
    # plt.savefig('confusion_matrixA.png')

    target_names = ['agriculture', 'commercial', 'harbor', 'idle_land', 'industrial', 'meadow', 'overpass', 'park', 'pond', 'residential', 'river', 'water']
    print(classification_report(y_true, y_pred_labels, target_names=target_names,digits=3))
