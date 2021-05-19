import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# a = tf.constant(1.)
# b = tf.constant(2.)
# print(a+b)
# print(tf.__version__)
# print('GPU:', tf.config.list_physical_devices('GPU'))
# print(tf.test.is_gpu_available())
# 查看gpu设备名称，出现/device:GPU:0说明安装成功了
tf.test.gpu_device_name()
# 查看gpu是否可用，结果为True说明可用
tf.test.is_gpu_available()
# 运行个张量常量，打印出Hello, TensorFlow说明安装成功
print(tf.constant('Hello, TensorFlow!'))

