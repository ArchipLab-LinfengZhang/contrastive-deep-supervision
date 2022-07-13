#Note that before running this command, you should first train a ResNet50 teacher on ImageNet.
python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:20401' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --resume checkpoint.pth.tar  /home/lthpc/datasets/ImageNet
