#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


# In[2]:


#исходное изображение графика
image = Image.open("48.png")
image


# In[3]:


sess = tf.Session()
# Построим три графика функции cross-entropy потерь
# Первый график
Y_pred = tf.linspace(-30., 7., 900)
# задаётся константа для графа вычислений
Y_label = tf.constant(1.)
Y_labels = tf.fill([900,], 1.)
#применяем операцию умножения 
x_entropy_vals = - tf.multiply(Y_label, tf.log(Y_pred)) - tf.multiply((1. - Y_label), tf.log(1. - Y_pred))
x_entropy_loss = sess.run(x_entropy_vals)
Y_array = sess.run(Y_pred)
# Второй график
Y_pred1 = tf.linspace(-50., 9., 500)
Y_label1 = tf.constant(1.)
Y_labels1 = tf.fill([500,], 2.)
x_entropy_vals1 = - tf.multiply(Y_label1, tf.log(Y_pred1)) - tf.multiply((1. - Y_label1+0.15), tf.log(1. - Y_pred1))
x_entropy_loss1 = sess.run(x_entropy_vals1)
Y_array1 = sess.run(Y_pred1)
# Третий график
Y_pred2 = tf.linspace(-50., 9., 1000)
Y_label2 = tf.constant(1.)
Y_labels2 = tf.fill([1000,], 2.)
x_entropy_vals2 = - tf.multiply(Y_label2+0.15, tf.log(Y_pred2)) - tf.multiply((1. - Y_label2+0.01), tf.log(1. - Y_pred2))
x_entropy_loss2 = sess.run(x_entropy_vals2)
Y_array2 = sess.run(Y_pred2)
# Сбор данных для построения графиков
plt.plot(Y_array2, x_entropy_loss2, "y", label = 'С')
plt.plot(Y_array, x_entropy_loss, label = 'A')
plt.plot(Y_array1, x_entropy_loss1, "purple", linestyle = "--", label = 'B')
plt.ylim(-1.5, 10)
# Легенда 
plt.legend()
# Название осей
plt.xlabel("Y") 
plt.ylabel("X")
# Заголовок графика
plt.title("Function quite looks like original")
plt.show()

