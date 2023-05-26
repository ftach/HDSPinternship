import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
import numpy as np
#from regularizers import implementation


class ForwardMCFA(Layer):
    ''' Create the MCFA sensing matrix Hm'''
    def __init__(self, input_dim=(512, 512, 31), noise=False, reg_param=0.5, dl=0.5, opt_H=True,
                 name=False, type_reg='sig', sig_param=10, shots=1, batch_size=1, snr=30, **kwargs):
        self.input_dim = input_dim
        self.shots = shots
        self.batch_size = batch_size
        self.noise = noise
        self.dl = dl
        self.Ld = int(self.input_dim[-1] * dl)
        self.opt_H = opt_H
        self.snr = snr
        self.type_reg = type_reg
        self.reg_param = reg_param
        self.sig_param = sig_param

        super(ForwardMCFA, self).__init__(name=name, **kwargs)

    def build(self, input_shape):

        if self.opt_H:
            print('Trainable MCFA')
            H_init = tf.constant_initializer(np.random.normal(0, 0.1, (1, self.input_dim[0], self.input_dim[0], self.Ld,
                                                                       self.shots)))
        else:
            print('Non-Trainable MCFA')

            H_init = tf.constant_initializer(
                np.random.randint(0, 2, size=(1, self.input_dim[0], self.input_dim[0], self.Ld,
                                              self.shots)))

            self.H = self.add_weight(name='H',
                                     shape=(1, self.input_dim[0], self.input_dim[0], self.Ld, self.shots),
                                     initializer=H_init, trainable=False)

    def call(self, inputs, **kwargs):

        H = self.H
        if self.type_reg == 'sig':
            H = tf.math.sigmoid(H * self.sig_param)
        L = self.input_dim[2]
        Ld = self.Ld
        M = self.input_dim[0]
        q = int(1 / self.dl)
        # Spatial decimator operator 
        kernel = np.zeros((1, 1, L, L // q))
        for i in range(0, L // q):
            kernel[0, 0, i * q:(i + 1) * q, i] = 1 / q

        input_im = tf.expand_dims(tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME'), -1) # f decimated 

        y = tf.expand_dims(tf.reduce_sum(tf.multiply(H, input_im), -2), -1)

        if self.noise:
            sigma = tf.reduce_sum(tf.math.pow(y, 2)) / ((M * M * self.batch_size) * 10 ** (self.snr / 10))
            y = y + tf.random.normal(shape=(self.batch_size, M, M, 1, 1), mean=0, stddev=tf.math.sqrt(sigma),
                                     dtype=y.dtype)

        return y, H



class TransposeMCFA(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, reg_param=0.5, dl=0.5, opt_H=True,
                 name=False, type_reg='sig', sig_param=10, shots=1, batch_size=1, snr=30, **kwargs):
        self.input_dim = input_dim
        self.shots = shots
        self.batch_size = batch_size
        self.noise = noise
        self.dl = dl
        self.Ld = int(self.input_dim[-1] * dl)
        self.opt_H = opt_H
        self.snr = snr
        self.type_reg = type_reg
        self.reg_param = reg_param
        self.sig_param = sig_param
        self.upsampling = tf.keras.layers.Conv2D(input_dim[-1],kernel_size=(1,1),padding='same')
        super(TransposeMCFA, self).__init__(name=name, **kwargs)



    def call(self, inputs, **kwargs):
        [y, H] = inputs


        L = self.input_dim[2]
        Ld = self.Ld
        M = self.input_dim[0]
        q = int(1 / self.dl)
        kernel = np.zeros((1, 1, L, L // q))
        for i in range(0, L // q):
            kernel[0, 0, i * q:(i + 1) * q, i] = 1 / q

        X = None
        for _ in range(Ld):
            if X is not None:
                X = tf.concat([X, y], 4)
            else:
                X = y
        X = tf.transpose(X, [0, 1, 2, 4, 3])

        X = tf.multiply(H, X)
        X = tf.reduce_sum(X, 4)
        X = X / tf.math.reduce_max(X)

        X = self.upsampling(X)
        return X

