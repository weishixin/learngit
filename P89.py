import tensorflow as tf
from numpy.random import RandomState
from numpy import  *
batch_size = 8

#两个输入节点
x = tf.placeholder( tf.float32 , shape=( None , 2) , name='x-input')
#回归问题一般只有一个输出节点
y_ = tf.placeholder( tf.float32 , shape=( None , 1) , name='y-input')

#定义了一个单层的神经网络前向传播过程，这里就是简单加权和
w = tf.Variable( tf.random_normal( [2,1] , stddev=1 , seed=1))
y = tf.matmul( x , w)
lambda1  = .5

# loss = tf.reduce_mean(tf.square(y_ - y )) + tf.contrib.layers.l2_regularizer(lambda1)(w)

weights = tf.constant( [[1.0,-2.0],[-3.0 , 4.0]] )
with tf.Session() as sess :
    #输出为(|1|+|-2|+|-3|+|4|) * 0.5 = 5 。其中0.5为正则化项的权重。
    print( sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    #输出为(1^2 +(-2)^2+(-3)^2+4^2)/2 * 0.5 = 7.5
    print( sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
    