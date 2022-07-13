import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
import PIL
from loss import *
from models.resnet import *
from cutout import *


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3, dim=1)
    softmax_targets = F.softmax(targets/3, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--model', default="resnet18", type=str)
parser.add_argument('--teacher', default="resnet50", type=str)
parser.add_argument('--teacher_path', default="resnet50.pth", type=str)
parser.add_argument('--supervision', default=True, type=bool)
args = parser.parse_args()
BATCH_SIZE = 128
LR = 0.1

transform1 = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                         transforms.RandomHorizontalFlip(), transforms.ToTensor(), Cutout(n_holes=1, length=16),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='data',
    train=True,
    download=False,
    transform=TwoCropTransform(transform1, transform2)
)
testset = torchvision.datasets.CIFAR100(
    root='data',
    train=False,
    download=False,
    transform=transform_test
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

if args.model == "resnet18":
    model_name = resnet18
    student_channel = 512
if args.model == "resnet50":
    model_name = resnet50
    student_channel = 2048
if args.model == "resnet101":
    model_name = resnet101
    student_channel = 2048
if args.model == "resnet152":
    model_name = resnet152
    student_channel = 2048

if args.teacher == 'resnet18':
    teacher_name = resnet18
    teacher_channel = 512
if args.teacher == 'resnet50':
    teacher_name = resnet50
    teacher_channel = 2048
if args.teacher == 'resnet101':
    teacher_name = resnet101
    teacher_channel = 2048
if args.teacher == 'resnet152':
    teacher_name = resnet152
    teacher_channel = 2048

net = model_name()
#   adaptation layer is used for the situations that
#   teachers and students have different sizes of projection head outputs
if teacher_channel != student_channel:
    net.adaptation_layer = nn.ModuleList([
        nn.Linear(student_channel, teacher_channel),
        nn.Linear(student_channel, teacher_channel),
        nn.Linear(student_channel, teacher_channel),
        nn.Linear(student_channel, teacher_channel)
    ])
net.to(device)
teacher = teacher_name()
teacher.to(device)
teacher.load_state_dict(torch.load(args.teacher_path))

criterion = nn.CrossEntropyLoss()
contra_criterion = SupConLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    epoch = 0
    for epoch in range(args.epoch):
        if epoch in [90, 180, 280]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        sum_c_loss = 0.0
        sum_kd_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            bsz = labels.size(0)
            #   inputs[0] -> origin, inputs[1] -> augmented
            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                t_outputs, t_feat_list = teacher(inputs)
            outputs, feat_list = net(inputs)
            outputs = outputs[:bsz]
            loss = criterion(outputs, labels)
            c_loss = 0
            kd_loss = 0
            for index in range(len(feat_list)):
                features = feat_list[index]
                t_features = t_feat_list[index]
                if student_channel != teacher_channel:
                    kd_loss += torch.dist(t_features, net.adaptation_layer[index](features), p=2) * 1e-1
                else:
                    kd_loss += torch.dist(t_features, features, p=2) * 1e-1
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                if args.supervision:
                    c_loss += contra_criterion(features, labels) * 1e-1
                else:
                    c_loss += contra_criterion(features) * 1e-1
            kd_loss += CrossEntropy(outputs, t_outputs[:bsz])
            loss += c_loss + kd_loss
            sum_c_loss += c_loss.item()
            sum_kd_loss += kd_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels.data).cpu().sum())
            if i % 50 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f Constrastive Loss: %.03f KD Loss: %.03f | Acc: %.4f%%'
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), sum_kd_loss / (i + 1), sum_c_loss / (i+1), 100 * correct / total))
        print("Waiting Test!")

        acc1 = 0
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += float(labels.size(0))
                correct += float((predicted == labels).sum())
            acc1 = (100 * correct/total)
            if acc1 > best_acc:
                best_acc = acc1
                torch.save(net.state_dict(), str(args.supervision)+args.model+"_student.pth")
        print('Test Set Accuracy: %.4f%%' % acc1)
    print("Training Finished, TotalEPOCH=%d" % args.epoch)
    print ("Best Accuracy", best_acc)



