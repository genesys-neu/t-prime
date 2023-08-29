import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(ResidualStack, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.pool_size = pool_size

    def forward(self, x):
        # 1*1 Conv Linear
        x = self.conv1(x)

        # Residual Unit 1
        shortcut = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + shortcut
        x = F.relu(x)

        # Residual Unit 2
        shortcut = x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = x + shortcut
        x = F.relu(x)

        # MaxPooling
        x = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

        return x


class ResNet(nn.Module):
    def __init__(self, iq_dim: int, num_samples: int, kernel_size: int, pool_size: int, num_classes: int):
        super(ResNet, self).__init__()
        num_kernels = 32

        self.res_stack1 = ResidualStack(in_channels=1, out_channels=num_kernels, kernel_size=(kernel_size, iq_dim),  pool_size=(pool_size, iq_dim))
        self.res_stack2 = ResidualStack(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(kernel_size,1),  pool_size=(pool_size,1))
        self.res_stack3 = ResidualStack(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(kernel_size,1),  pool_size=(pool_size,1))
        self.res_stack4 = ResidualStack(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(kernel_size,1),  pool_size=(pool_size,1))
        self.res_stack5 = ResidualStack(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(kernel_size,1),  pool_size=(pool_size,1))
        self.res_stack6 = ResidualStack(in_channels=num_kernels, out_channels=num_kernels, kernel_size=(kernel_size,1),  pool_size=(pool_size,1))

        self.flatten = nn.Flatten()

        rand_x = torch.Tensor(np.random.random((1, num_samples, iq_dim)))
        rand_x = rand_x.unsqueeze(1)
        rand_x = self.res_stack1(rand_x)
        rand_x = self.res_stack2(rand_x)
        rand_x = self.res_stack3(rand_x)
        rand_x = self.res_stack4(rand_x)
        rand_x = self.res_stack5(rand_x)
        rand_x = self.res_stack6(rand_x)
        rand_x = self.flatten(rand_x)

        self.fc1 = nn.Linear(rand_x.numel(), 128)
        self.alpha_dropout = nn.AlphaDropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

        self.norm = nn.LayerNorm([num_samples, iq_dim])

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.norm(x)
        x = self.res_stack1(x)
        x = self.res_stack2(x)
        x = self.res_stack3(x)
        x = self.res_stack4(x)
        x = self.res_stack5(x)
        x = self.res_stack6(x)

        x = self.flatten(x)
        x = F.selu(self.fc1(x))
        x = self.alpha_dropout(x)
        x = self.fc2(x)

        return x

"""

# NOTE: model reproduced from here: https://github.com/liuzhejun/ResNet-for-Radio-Recognition/blob/master/ResNet_incomplete_dataset.ipynb


def residual_stack(Xm,kennel_size,Seq,pool_size):
    #1*1 Conv Linear
    Xm = Conv2D(32, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    #Residual Unit 1
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv2", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #Residual Unit 2
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv4", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    X = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv5", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm


in_shp = X_train.shape[1:]   #每个样本的维度[1024,2]
#input layer
Xm_input = Input(in_shp)
Xm = Reshape([1,1024,2], input_shape=in_shp)(Xm_input)
#Residual Srack
Xm = residual_stack(Xm,kennel_size=(3,2),Seq="ReStk0",pool_size=(2,2))   #shape:(512,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk1",pool_size=(2,1))   #shape:(256,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk2",pool_size=(2,1))   #shape:(128,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk3",pool_size=(2,1))   #shape:(64,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk4",pool_size=(2,1))   #shape:(32,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk5",pool_size=(2,1))   #shape:(16,1,32)

#############################################################################
#      多次尝试发现减少一层全连接层能使loss下降更快
#      将AlphaDropout设置为0.3似乎比0.5效果更好
#############################################################################
#Full Con 1
Xm = Flatten(data_format=data_format)(Xm)
Xm = Dense(128, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.3)(Xm)
#Full Con 2
Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense2")(Xm)
#SoftMax
Xm = Activation('softmax')(Xm)
#Create Model
model = Model.Model(inputs=Xm_input,outputs=Xm)
"""