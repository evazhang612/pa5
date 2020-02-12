######################################################################
# Author: Winnie Lin
# Tensorflow implementation of 2D-FAN.
######################################################################
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

def batchnorm(name):
    return layers.BatchNormalization(name=name,axis=-1,epsilon=1e-05,momentum=0.9)

class ConvBlock(keras.Model):
    def __init__(self, i_dim, o_dim,**kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.i_dim=i_dim
        self.o_dim=o_dim

        self.layer1=keras.Sequential([
                batchnorm("bn1"),
                layers.ReLU(),
                layers.Conv2D(o_dim//2,kernel_size=(3,3),strides=1,padding="same",use_bias=False,name="conv1")
                ],name="block1")
        self.layer2=keras.Sequential([
                batchnorm("bn2"),
                layers.ReLU(),
                layers.Conv2D(o_dim//4,kernel_size=(3,3),strides=1,padding="same",use_bias=False,name="conv2")
                ],name="block2")
        self.layer3=keras.Sequential([
                batchnorm("bn3"),
                layers.ReLU(),
                layers.Conv2D(o_dim//4,kernel_size=(3,3),strides=1,padding="same",use_bias=False,name="conv3")
            ],name="block3")

        if i_dim!=o_dim:
            self.layer_ds=keras.Sequential([
                    batchnorm("0"),
                    layers.ReLU(),
                    layers.Conv2D(o_dim,kernel_size=(1,1),strides=1,padding="same",use_bias=False,name="2")
                ],name="downsample")

    def call(self,x):
        o1=self.layer1(x)
        o2=self.layer2(o1)
        o3=self.layer3(o2)
        if self.i_dim==self.o_dim:
            res=x
        else:
            res=self.layer_ds(x)
        return tf.add(tf.concat([o1,o2,o3],axis=-1),res)

class HourGlass(keras.Model):
    def __init__(self, depth, o_dim,**kwargs):
        super(HourGlass, self).__init__(**kwargs)
        self.depth = depth
        self.o_dim = o_dim

        self.b1 = [ConvBlock(o_dim,o_dim,name="b1_%d"%(depth-i)) for i in range(depth)]
        self.b2 = [ConvBlock(o_dim,o_dim,name="b2_%d"%(depth-i)) for i in range(depth)]
        self.b3 = [ConvBlock(o_dim,o_dim,name="b3_%d"%(depth-i)) for i in range(depth)]
        self.pool = [layers.AveragePooling2D(pool_size=(2,2)) for i in range(depth)]
        self.up = [layers.UpSampling2D(size=(2,2)) for i in range(depth)]
        self.b2_plus = ConvBlock(o_dim,o_dim,name="b2_plus_1")

    def call(self,x):

        lows=[x]

        for i in range(self.depth):
            curr=self.pool[i] (lows[-1])
            curr=self.b2[i] (curr)
            if i==self.depth-1:
                curr=self.b2_plus(curr)
            else:
                lows.append(curr)

        for i in range(self.depth-1,-1,-1):
            top =self.b1[i] (lows[i])
            bot =self.b3[i] (curr)
            bot =self.up[i] (bot)
            curr=tf.add(top,bot)

        return curr

class FAN(keras.Model):
    def __init__(self, n_modules=1):
        super(FAN, self).__init__()
        self.n_modules=n_modules

        self.base = keras.Sequential(
            [layers.Conv2D(64,kernel_size=7,strides=2,padding="valid",name="conv1"),
             batchnorm("bn1"),
             layers.ReLU(),
             ConvBlock(64,128,name="conv2"),
             layers.AveragePooling2D(),
             ConvBlock(128,128,name="conv3"),
             ConvBlock(128,256,name="conv4"),
             ],
             name="base"
            )

        self.hgs=[keras.Sequential(
                    [HourGlass(4,256,name="m%d"%i),
                     ConvBlock(256,256,name="top_m_%d"%i),
                     layers.Conv2D(256,kernel_size=1,strides=1,padding="valid",name="conv_last%d"%i),
                     batchnorm("bn_end%d"%i),
                     layers.ReLU()
                    ],name="hg_%d"%i) for i in range(n_modules)]

        self.ls      =[layers.Conv2D(68,kernel_size=1,strides=1,padding="valid",name="l%d"%i)
                        for i in range(n_modules)]

        self.split_as=[layers.Conv2D(256,kernel_size=1,strides=1,padding="valid",name="al%d"%i)
                        for i in range(n_modules-1)]
        self.split_bs=[layers.Conv2D(256,kernel_size=1,strides=1,padding="valid",name="bl%d"%i)
                        for i in range(n_modules-1)]

    def call(self,x):
        x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], "CONSTANT")
        x=self.base(x)
        previous=x
        outputs=[]
        for i in range(self.n_modules):
            ll =self.hgs[i] (previous)
            out=self.ls[i] (ll)
            outputs.append(out)
            if i<self.n_modules-1:
                ll=self.split_bs[i] (ll)
                tp=self.split_as[i] (out)
                previous=previous+ll+tp
        return outputs
