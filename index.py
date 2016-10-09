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

    def conv_layer(hidden_layer_output, name):
           h0 = tf.nn.relu(input_layer(hidden_layer_output, FILTERS[0], stride=1, name=name+'_h0_conv'))
           h1 = tf.nn.relu(input_layer(h0, filters[1], stride=2, name=name+'_h1_conv'))
           h2 = tf.nn.relu(input_layer(h1, filters[2], stride=1, name=name+'_h2_conv'))
           h3 = tf.nn.tanh(input_layer(h2, filters[3], stride=1, name=name+'_h3_conv'))

    def gen_data(n=BATCH_SIZE,msg_len=MSG_LEN,key_len=KEY_LEN):
        return(np.random.randint(0,2,size=(n,msg_len))*2-1), \
                                                             (np.random.randint(0,2,size=(n,key_len))*2-1)
    def init_weights(name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layer, xavier_initializer())
    def build_model():
        self.w_one=init_weights("w_one",[2* self.N, 2*self.N])
        self.w_one=init_weights("w_two",[2* self.N, 2*self.N])
        self.w_one=init_weights("police_w1",[2* self.N, 2*self.N])
        self.w_one=init_weights("police_w2",[2* self.N, 2*self.N])
        
        




app=Encrypt_AI()
