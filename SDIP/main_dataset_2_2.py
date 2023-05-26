import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.python.framework import ops
from models.main import *  # donde esta el modelo y las funciones necesarias como psnr y demas
import scipy.io
import tensorflow as tf
from scipy.sparse import csr_matrix, find
import time
from os import listdir
from os.path import isfile,join
import random
print(tf.__version__)

def set_seed(seed: int = 42) -> None:
  """Memorize init to keep results consistent between runs"""
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  tf.compat.v1.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*9)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)




from models.cassi import ForwardCASSI
from models.mcfa import  ForwardMCFA




m = 512
n= 512
l = 31
shots_hs = 4
shots_ms = 4
lr = 1e-3


results_folder = 'test'
try:
    os.mkdir(results_folder)
except OSError as error:
    print(error)

total_shots = shots_ms+shots_hs
iters = 5000 # 15000 actually


# Load Image
imgs = {'kaist': 'img','cave':'im'}
im = 'kaist'
Mat= scipy.io.loadmat('test_'+ im +'.mat')

# testSI=np.double(Mat['im'])
testSI=np.double(Mat[imgs[im]])
testSI=testSI/np.max(testSI)
RGB = testSI[:,:,(25, 22, 11)]

# Define operators 
F = [] # whole sensing matrix
y = []
for i in range(1, shots_hs + 1):
    F.append(
        ForwardCASSI(input_dim=(m, n, l), noise=False, bin_param=0.5, opt_H=False, name='Sensing_hs_' + str(i), shots=1,
                     ds=0.5, batch_size=1, snr=30))
    # temp, _ = F[i - 1](tf.expand_dims(testSI, 0))
    # H_hs.append(H_temp)
    # y.append(temp)

for i in range(1, shots_ms + 1):
    t = ForwardMCFA(input_dim=(m, n, l), noise=False, opt_H=False, name='Sensing_ms_' + str(i), shots=1, dl=0.5,
                    batch_size=1, snr=30)
    F.append(t)
    # temp, _ = t(tf.expand_dims(testSI, 0))
    # y.append(temp)

for i in range(total_shots):
    y_temp = F[i](tf.expand_dims(testSI, 0))[0]
    y.append(y_temp)





shots_per_eval = 5 # MODIFY 

path = f'{results_folder}/init_sms{shots_ms}_shs_{shots_hs}_se_{shots_per_eval}_im'

try:
    os.mkdir(path)
except OSError as error:
    print(error)
input_x = np.zeros(shape=(1, m, n, l))

rho = 0.4

# Visualization Parameters setup
Freq = 20

# Optimization
model = UNetL(input_size=(m, n, l), L=l, fact=rho, F=F, shots_hs=shots_hs, shots_ms=shots_ms, y=y,
                    shots_per_eval=shots_per_eval)
optimizad = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.99, amsgrad=False)

model.compile(optimizer=optimizad)
start = time.time()

h = model.fit(tf.expand_dims(testSI, 0), tf.expand_dims(testSI, 0), epochs=iters, batch_size=1,
              verbose=1) # testSI twice because we don't need a ground truth for the training (but tensorflow does)
end = time.time()
duration = end - start
a = model(tf.expand_dims(testSI, 0))
PSNR_Final = h.history['psnr']
loss = h.history['loss']
a = np.array(np.squeeze(a))
# Convergence Curve
# PSNRs = model.PSNRs

# Low-Rank Tucker Representation of tensor Z
func = K.function([model.layers[0].input], [model.layers[1].output])
ZTuckerRepr = func(np.zeros(shape=(1, m, n, l)))
ZTuckerRepr = np.asarray(ZTuckerRepr).reshape((m, n, l), order="F")

VisualGraphs(a, a, ZTuckerRepr, PSNR_Final, testSI, [25, 22, 11],path=path)

scipy.io.savemat(path+'/metrics.mat',{'psnr':PSNR_Final,'time':duration, 'loss':loss,'rec':a})
