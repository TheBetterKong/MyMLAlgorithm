#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project  ：MyMLAlgorithm 
# @Date     ：2021/6/18 15:20
# @Author   : TheBetterKong
# @Site     : 
# @File     : simple_linear_regression.py
# @Software : PyCharm

import tensorflow as tf
import matplotlib.pyplot as plt


# 一些超参数
TRUE_W = 3.0            # 真实线条的 w
TRUE_B = 2.0            # 真实线条的 b
NUM_EXAMPLE = 1000      # 生成的样本数
EPOCHS = 10

# 全局变量
Ws, bs = [], []


def generate_data():
    x = tf.random.normal(shape=[NUM_EXAMPLE])
    noise = tf.random.normal(shape=[NUM_EXAMPLE])
    y = x * TRUE_W + TRUE_B + noise
    return x, y


class LinearModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b


def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def train(model : LinearModel, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    dw, db = t.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


def train_loop(model : LinearModel, x, y):
    for epoch in range(EPOCHS):
        train(model, x, y, learning_rate=0.1)

        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
              (epoch, Ws[-1], bs[-1], current_loss))


if __name__ == "__main__":
    x, y = generate_data()
    model = LinearModel()

    print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" %
              (model.w, model.b, loss(y, model(x))))

    train_loop(model, x, y)

    plt.plot(range(EPOCHS), Ws, "r",
             range(EPOCHS), bs, "b")
    plt.plot([TRUE_W] * EPOCHS, "r--",
             [TRUE_B] * EPOCHS, "b--")
    plt.legend(["w", "b", "True W", "True b"])
    plt.show()
