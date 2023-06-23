''' Run different tests changing parameters values.'''
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import time
import h5py

class RandomInit(Layer):
    def __init__(self, d=256, M=256):
        super(RandomInit, self).__init__()

        self.d = d
        self.M = M
        self.dense = tf.keras.layers.Dense(M**2,use_bias=True,activation='relu')
    def build(self, input_shape):
        self.z = self.add_weight(shape=(1,self.M,self.M,1),trainable=True,initializer='random_normal')
    def call(self, inputs):
        #z = tf.random.normal(shape=(1,self.M,self.M,1))*0.01
        #z = z/tf.reduce_max(z)
        #out = self.dense(z)
        #out = tf.reshape(out,[1,self.M,self.M,1])
        return self.z
    
def fun_PSNR(img,res):

    [M,N,L]=img.shape
    temp=1./(M*N*L)*np.sum(np.power(img-res,2))
    psnr= 10*np.log10(np.max(np.power(img,2)/temp))
    return psnr    

def UNetCT(pretrained_weights=None, input_size=(256, 256, 1), L=1, d=500,F=0,
          shots_per_eval=1, y=None, total_shots=10):
    L_2 = 2 * L
    L_3 = 3 * L
    L_4 = 4 * L
    inputs = Input(input_size)
    #inicial = XoLayer(largo=input_size[0], ancho=input_size[1], profun=1, fact=0.4)(inputs)
    inicial = RandomInit(M=input_size[0],d=d)(inputs)
    #conv1 = Dropout(0.2)(inputs)
    conv1 = Conv2D(input_size[-1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inicial)
    conv1 = Conv2D(input_size[-1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    b_norm5 = BatchNormalization()(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    b_norm6 = BatchNormalization()(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    b_norm7 = BatchNormalization()(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b_norm7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    conv8 = Conv2D(input_size[-1], 3, padding='same', activation='sigmoid')(conv7)

    model = Model(inputs, conv8)
    #model.summary()

    loss = 0
    
    eval_idx = np.random.permutation(total_shots)[:shots_per_eval]
    for i in eval_idx:
        loss += tf.reduce_mean(tf.square(y[i] - F[i](conv8)))
    loss = loss / shots_per_eval
    model.add_loss(loss)
    ssim = tf.image.ssim(conv8, inputs, max_val=1.0)
    model.add_metric(ssim, name='ssim', aggregation='mean')
    psnr = tf.image.psnr(inputs, conv8, 1)
    model.add_metric(psnr, name='psnr', aggregation='mean')


    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

class image_prj(tf.keras.layers.Layer):
    """
    Image projection layer made of Wavelet and Radon transform.
    prj_angles: a list of angles
    """

    def __init__(self, theta=(0, 90, 180), M=512,S=512,**kwargs):
        self.theta = theta
        self.M = M
        self.S = S
        super(image_prj, self).__init__(**kwargs)

    def build(self, input_shape):
        super(image_prj, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, image_tensor):

        # Reshape the image tensor as batch, height, width, channel
        image_tensor_batched = tf.expand_dims(image_tensor, axis=1)
        image_tensor_batched = tf.expand_dims(image_tensor, axis=-1)    
        input_shape = (self.M, self.S)

        low_tensor_batched, high_tensor_batched = self.dwt(image_tensor_batched) # compute wavelet decomposition
        tr_high_tensor_batched = self.tf_hard_threshold(high_tensor_batched) # threshold high frequency coefficients
        input_shape = (tr_high_tensor_batched.shape[1], tr_high_tensor_batched.shape[2]) # input shape changes after wavelet decomposition

        # APPLY RADON TRANSFORM
        tr_high_tensor = tf.reshape(tr_high_tensor_batched, input_shape)
        padded_image_tensor, center = self.prepare_img(tr_high_tensor, input_shape) # get padded image tensor
        A = self.radon_transfom(padded_image_tensor, center) # compute Radon transform 

        return A
    
    def dwt(self, image_tensor_batched):
        ''' Perfrom a 2D discrete wavelet transform on the input data. 
        Parameters
        ----------
        image_tensor_batched: tf.tensor 
            Input data of shape (batch, height, width, channel) to be transformed
        Returns
        -------
        low_pass: tf.tensor 
            Low pass of shape (batch, height, width, channel) component of the wavelet transform
        high_pass: tf.tensor
            High pass of shape (batch, height, width, channel) component of the wavelet transform
        '''

        # Haar wavelet filter 
        low_wavelet_filter = tf.constant([[0.1601, 0.6038, 0.7243, 0.1384]], dtype=tf.float32)
        low_wavelet_filter = tf.reshape(low_wavelet_filter, [2, 2, 1, 1])
        high_wavelet_filter = tf.constant([[-0.1384, 0.7243, -0.6038, 0.1601]], dtype=tf.float32)
        high_wavelet_filter = tf.reshape(high_wavelet_filter, [2, 2, 1, 1])

        # Wavelet decomposition 
        low_pass = tf.nn.conv2d(image_tensor_batched, low_wavelet_filter, strides=[1, 2, 2, 1], padding='SAME')
        high_pass = tf.nn.conv2d(image_tensor_batched, high_wavelet_filter, strides=[1, 2, 2, 1], padding='SAME')

        return low_pass, high_pass
    
    def prepare_img(self, image_tensor, input_shape):
        ''' Pad image to avoid loose of information during Radon transform. 
        Parameters
        -------
        image_tensor: tf.tensor 
            Input data of shape (height, width) to be transformed
        input_shape: tuple
            Shape (height, width)
        Returns
        -------
        padded_image_tensor: tf.tensor 
            Padded image tensor of shape (batch, height, width, channel)
        center: int 
            Padded image center position 
        '''
        # Get dimensions
        diagonal = np.sqrt(2) * max(input_shape)
        pad = [int(np.ceil(diagonal - s)) for s in input_shape]
        new_center = [(s + p) // 2 for s, p in zip(input_shape, pad)]
        old_center = [s // 2 for s in input_shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        pad_width = tf.constant(pad_width)

        # Pad image 
        padded_image = tf.pad(image_tensor, pad_width, mode='constant', constant_values=0)
        if padded_image.shape[0] != padded_image.shape[1]:
            raise ValueError('padded_image must be a square')
        
        # Prepare image for Radon transform
        center = padded_image.shape[0] // 2
        padded_image_tensor = tf.expand_dims(padded_image, axis=0)
        padded_image_tensor = tf.expand_dims(padded_image_tensor, axis=-1)

        return padded_image_tensor, center 
    
    def tf_hard_threshold(self, image_tensor, threshold=0.075):
        ''' Apply hard thresholding to the input data.
        Parameters
        ----------
        image_tensor: tf.tensor
            Input data of shape (batch, height, width, channel) to be thresholded
        threshold: float
            Threshold value
        Returns
        -------
        thresholded_tensor: tf.tensor
            Thresholded data of shape (batch, height, width, channel)
        '''
        mask = tf.abs(image_tensor) >= threshold
        thresholded_tensor = tf.where(mask, image_tensor, tf.zeros_like(image_tensor))

        return thresholded_tensor
    
    def radon_transfom(self, padded_image_tensor, center):
        ''' Compute Radon transform of the input data.
        Parameters
        ----------
        padded_image_tensor: tf.tensor
            Input data of shape (batch, height, width, channel) to be transformed
        center: int
            Padded image center position
        Returns
        -------
        A: tf.tensor
            Sensing matrix made of wavelet decomposition and Radon transform
        '''

        radon_image = tf.zeros((padded_image_tensor.shape[1], padded_image_tensor.shape[2]), dtype=tf.float32)
        A = []
        for i, angle in enumerate(self.theta):  # Here theta must be in radians
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            R = tf.constant([cos_a, sin_a, -center * (cos_a + sin_a - 1),
                             -sin_a, cos_a, -center * (cos_a - sin_a - 1),
                             0., 0.], dtype=tf.float32)
            R = tf.reshape(R, (1, 8))
            rotated = tf.raw_ops.ImageProjectiveTransformV3(images=padded_image_tensor, transforms=R, interpolation="BILINEAR",
                                                            output_shape=radon_image.shape, fill_mode="WRAP",
                                                            fill_value=0.0)

            radon_line = tf.reduce_sum(tf.squeeze(rotated), axis=0)
            A.append(tf.squeeze(radon_line))
        A = tf.transpose(A[:][:])

        return A/tf.reduce_max(A)

def prj(input_size=(362, 362), angles=(0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)):
    ''' Creates the Wavelet x Radon transform model
    Parameters
    ----------
    input_size: tuple
        Input image size
    angles: tuple
        Radon transform angles in radians
    Returns
    -------
    model: tf.keras.Model
        Wavelet x Radon transform model
    '''

    img_in = tf.keras.Input(input_size)
    prj_out = image_prj(theta=angles,M=input_size[0], S=input_size[1])(img_in)
    model = tf.keras.Model(inputs=img_in, outputs=prj_out)
    #model.summary()

    return model


results = pd.DataFrame(columns=['img_nbr', 'L', 'B', 'max_PSNR', 'max_SSIM', 'Time'])
all_img = []
# Prepare images 
img_data = h5py.File('ground_truth_test_000.hdf5', 'r')
img_dset = tf.image.resize(tf.expand_dims(img_data['data'],-1),[512,512])
index = [1, 4, 9, 14, 20]
for i in index:
    image = img_dset[i, :, :]
    image = np.transpose(image)
    image = np.float32(image / np.max(image))
    all_img.append(image)

def get_model_inputs(image, S, NUM_ANGLES=30):
    '''
    Parameters
    ----------
    image: tf.tensor
        Input image of shape (batch, height, width, channel)
    S: int
        Total number of shots
    NUM_ANGLES: int
        Number of angles for the Radon transform

    Returns
    -------
    F: list
        List of Fourier transform of the input image
    y: list
        List of Radon transform of the input image
    theta: list
        List of Radon transform angles
    theta_rad: list
        List of Radon transform angles in radians
    '''
    _, sz_x, sz_y = image.shape
    F = []
    y =[]
    theta = []
    theta_rad = []
    for i in range(S) :
        angles = np.random.normal(loc=90, scale=90, size=NUM_ANGLES)
        rad_angles = [(x* np.pi / 180) for x in angles] 
        angles = sorted(angles)
        rad_angles = sorted(rad_angles)
        theta.append(angles) 
        theta_rad.append(rad_angles)
        F.append(prj((sz_x, sz_y), theta_rad[i]))
        y.append(F[i](image))

    return F, y, theta, theta_rad

def train_model(L, B, S, F, y, input_shape, lr=0.001, iters=10000): 
    '''
    Parameters
    ----------
    L: int
        Number of layers
    B: int
        Number of shots per evaluation
    S: int
        Total number of shots
    F: list
        List of forward operators
    y: list
        List of measurements 
    input_shape: tuple
        Input image shape (height, width)
    lr: float
        Learning rate
    iters: int
        Number of iterations
    Returns
    -------
    model: tf.keras.Model
        Trained model
    h: tf.keras.callbacks.History
        History of the training
    duration: float
        Training time in seconds
    '''

    model = UNetCT(input_size=(input_shape[0],input_shape[1],1), F=F, y=y, L=L, shots_per_eval=B,d=400, total_shots=S)
    optimizad = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.99, amsgrad=False)
    model.compile(optimizer=optimizad, loss='mean_squared_error')
    start = time.time()
    h = model.fit(x=image,y=image, epochs=iters, batch_size=1, verbose=0)
    end = time.time()
    duration = end - start

    return model, h, duration 
results.to_csv('results.csv', index=False) 
img_nbr = 0
all_L = [4, 16, 32]
all_B = range(2, 9, 2)
all_S = [10, 20, 30]
input_shape = (512, 512)
for image in all_img:
    img_nbr += 1
    for L in all_L: 
        for B in all_B:
            for S in all_S: 
                F, y, theta, theta_rad = get_model_inputs(image, S)
                model, h, duration = train_model(L, B, S, F, y, input_shape)
                reconstruction = tf.squeeze(model(image)).numpy()
                plt.imsave('reconstruction'+str(img_nbr)+'.png', reconstruction, cmap='gray')
                PSNR_Final = h.history['psnr']
                SSIM_Final = h.history['ssim']
                loss = h.history['loss']
                max_PSNR = np.max(PSNR_Final)
                max_SSIM = np.max(SSIM_Final)
                df2 = pd.DataFrame({'img_nbr': [img_nbr], 'L': [L], 'B': [B], 'max_PSNR': [max_PSNR], 'max_SSIM': [max_SSIM], 'Time': [duration]})
                results = pd.concat([results, df2], ignore_index=True)

results.to_csv('results.csv', index=False) 