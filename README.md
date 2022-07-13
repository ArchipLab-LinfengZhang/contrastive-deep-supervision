# Contrastive Deep Supervision

This is the code for contrastive deep supervision and distilled contrastive deep supervision.

##### Install. 

Install the based packages for training.

```
pip install torch torchvision
```

##### Contrastive Deep Supervision on CIFAR

```
python train.py --model=$model name$ 
```

$model name$ is the choice of student models,  including [resnet18 | resnet50 | resnet101 | resnet152 ]

##### Distilled Contrastive Deep Supervision on CIFAR

Before applying distilled contrastive deep supervision, you should first train a teacher model with contrastive deep supervision. Taking ResNet152 teacher as an example, you should run

```
python train.py --model=resnet152 
```

Then, train the students with the following script.

```
python distill.py --model=$student name$ --teacher=$teacher name$ --teacher_path=$teacher checkpoint path$
```

$student name$ is the choice of student models,  including [resnet18 | resnet50 | resnet101 | resnet152 ].  $teacher name$ is the choice of teacher models,  including [resnet18 | resnet50 | resnet101 | resnet152].  $teacher checkpoint path$ is the path of teacher checkpoint. Note that the teacher should be trained with contrastive deep supervision.

**Experiments on ImageNet**

Please refer to the run.sh file in the folder to perform contrastive deep supervision and distilled contrastive deep supervision on ImageNet experiments. Note that you should train a teacher model before applying knowledge distillation.
