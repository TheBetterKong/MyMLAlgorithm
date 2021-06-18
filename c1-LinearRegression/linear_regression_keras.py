#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project  ：MyMLAlgorithm 
# @Date     ：2021/6/18 16:01
# @Author   : TheBetterKong
# @Site     : 
# @File     : linear_regression_keras.py
# @Software : PyCharm
from abc import ABC

import tensorflow as tf


# 一些超参数
TRUE_W = 3.0            # 真实线条的 w
TRUE_B = 2.0            # 真实线条的 b
NUM_EXAMPLE = 1000      # 生成的样本数


def generate_data():
    x = tf.random.normal(shape=[NUM_EXAMPLE])
    noise = tf.random.normal(shape=[NUM_EXAMPLE])
    y = x * TRUE_W + TRUE_B + noise
    return x, y


class LinearModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x, **kwargs):
        return self.w * x + self.b


if __name__ == "__main__":
    x, y = generate_data()
    keras_model = LinearModel()
    keras_model.compile(
        run_eagerly=False,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.mean_squared_error
    )

    print(tf.shape(x))
    keras_model.fit(x, y, epochs=10, batch_size=1000)
