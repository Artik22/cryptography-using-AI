import tensoflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


class Encrypt_AI():
    def __init__(self, *args, **kwargs):
        self.author="Ben Nguru"
    def __start__encrypt():
        filters=[
            [4, 1, 2],
            [2, 2, 4],
            [1, 4, 4],
            [1, 4, 1]]
        MSG_LEN = 16
        KEY_LEN = 16
        BATCH_SIZE = 512
        NUM_EPOCHS = 60
        LEARNING_RATE = 0.0008

    def input_layer(input_, filter_shape, stride, name="input_layer"):
        with tf.variabe_scope(name):
            w=tf.get_variable('w'. shape=filter_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
            conv=tf.nn.input_layer(input_,w,stride, padding="SAME")
            return conv

 
        
        




app=Encrypt_AI()
