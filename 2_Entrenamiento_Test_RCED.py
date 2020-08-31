# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:57:50 2020

@author: Paco
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import tensorflow as tf
from sklearn.model_selection import train_test_split

n_mels=128
tiempo=196

X_train=[]
Y_train=[]

#convertimos a mels, convertimos a escala logarítmica de -80..+80db, y normalizamos con rango -1 a 1
def preprocess(wave):
    array = librosa.feature.melspectrogram(y=wave, n_mels=n_mels)
    array = librosa.power_to_db(array)
    array=array/80
    return array

#desnormalizamos, calculamos la inversa de mel y convertimos a formato de onda
def postprocess(array):
    array=array*80
    array=librosa.db_to_power(array)
    wave = librosa.feature.inverse.mel_to_audio(array, n_iter=32)
    return wave

#cambiamos el array para que contenga 4 dimensiones para las convolucionales 2D: batch, dim1, dim2, canales
def preshape(array):
    array= array.reshape(1,n_mels,tiempo,1)
    return array

#cambiamos el array a 2 dimensiones para poder hacer la inversa de mel
def postshape(array):
    array= array.reshape(n_mels,tiempo)
    return array

#padding de ceros a la derecha del audio hasta llegar a 100000
#el efecto real es añadir silencio al final del audio, la utilidad es para que las muestras midan igual.
def rpad(wave,n_pad=100000):
    dif=n_pad-len(wave)
    wave=np.concatenate((wave, np.zeros(dif,dtype=np.float32)), axis=None)
    return wave

#esta función predice un audio a través de otro
def predictTest():   
    voz1_data, voz1_sr = librosa.load('male.mp3', sr=11025)  # time series data,sample rate
    voz1_data=rpad(voz1_data)
    X1=preshape(preprocess(voz1_data))

    print("PREDICIENDO..")
    Ypred1 = model.predict(X1)

    print("MELAUDIO..")
    voz2_data = postprocess(postshape(Ypred1))   
    T = voz2_data
    #para guardan como wave tenemos que convertir los datos del audio a enteros
    Tint = T / np.max(T) * 32767

    wavfile.write("reconstructionRCED.wav", voz1_sr, Tint.astype('int16'))
    print("WAV")


basepath='c:/VOICE_es'


#DESCOMENTAR

#X_train=np.load('X.npy') #comentar cuando no dispongamos de los datos precargados
#Y_train=np.load('Y.npy')

X_train=np.load('XAA.npy') #comentar cuando no dispongamos de los datos precargados
Y_train=np.load('YAA.npy')


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)


print("MEL PRECARGADOS")

## import keras modules
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ZeroPadding2D, Activation
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


numFeatures  = 128
numSegments  = 196


def build_model(l2_strength):
    #Entrada compuesta por 128 mels, 196 segmentos y 1 canal
    inputs = Input(shape=[numFeatures, numSegments, 1])
    
    x = inputs

    # Añadimos un pequeño padding para que coincida correctamente las dimensiones de la capa de input y la de output
    # -----
    x = ZeroPadding2D(((2,2), (4,0)))(x)
    
    # Utilizamos una combinación de parámetros y filtros convolucionales 2d + activación Relu + batchNorm de acuerdo al paper
    # "A FULLY CONVOLUTIONAL NEURAL NETWORK FOR SPEECH ENHANCEMENT" (2016)
    
    # Filtros: 10-12-14-15-19-21-23-25-23-21-19-15-14-12-10-1
    
    # No se utiliza MaxPooling ni técnicas de downsampling ya que afectan a la pixel-wise-habilidad, pierden demasiada información.
    # -----
    
    #Comenzamos en las capas extremas con kernel 5,5 para captar zonas más grandes, disminuyendo a 3,3 en el resto
    
    #Usamos regularización la L2 sugerida en el Paper.
    x = Conv2D(filters=10, kernel_size=[5, 5], strides=[1, 1], padding='valid', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    #Unimos de forma directa con las capas de abajo (skip), lo cual: 
    #evita el gradient vanishing, acelera el entrenamiento, y permite una gran profundidad de capas 
    #ya que las features pueden transmitirlas a las capas de abajo

    #link 0 
    skip0 = Conv2D(filters=12, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip0)    

    x = BatchNormalization()(x)

    x = Conv2D(filters=14, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # -----
    x = Conv2D(filters=15, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    
    #link 1
    skip1 = Conv2D(filters=19, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip1)
    x = BatchNormalization()(x)

    x = Conv2D(filters=21, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=23, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=25, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=23, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=21, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=19, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip1 
    #Añadimos el link 1
    
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=15, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=14, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=12, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip0
    # Añadimos el link 0
    
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=10, kernel_size=[5, 5], strides=[1, 1], padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Añadimos 0.2 de Dropout 2D.. valores de hasta 0.5 funcionan bien 
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    
    # Capa de salida, 1 filtro de 128,196,1 y kernel recomendado 3,3
    x = Conv2D(filters=1, kernel_size=[3,3], strides=[1, 1], padding='same')(x)

    model = Model(inputs=inputs, outputs=x)

    #parámetros optimizer de "A FULLY CONVOLUTIONAL NEURAL NETWORK FOR SPEECH ENHANCEMENT"
    optimizer = tf.keras.optimizers.Adam(lr=0.0015,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    #pesos despues de pretrain
    #optimizer = tf.keras.optimizers.Adam(lr=0.00001,beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


model = build_model(l2_strength=0.00001)

print("COMPILADO")

plot_model(model, to_file='model_RCED.png', show_shapes=True, show_layer_names=True)
#tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)



model.summary()

# fit the model
print("FIT START")

cbreducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                              patience=4, min_lr=0.00000001)

early_stopping_callback = EarlyStopping(monitor='val_mae', patience=100, restore_best_weights=True)

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
checkpoint_callback = ModelCheckpoint(filepath='rced1.h5', monitor='val_mae', save_best_only=True)



#DESCOMENTAR SI HEMOS PREENTRENADO
#model.load_weights('rced2a_01121.h5')



history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), shuffle=True,
                    epochs=300, batch_size=5, #  steps_per_epoch=600,
                    verbose=1, callbacks=[early_stopping_callback,cbreducelr,checkpoint_callback])

print("FIT END")

#Check how loss & mse went down
epoch_loss = history.history['loss']
epoch_val_loss = history.history['val_loss']
epoch_mae = history.history['mae']
epoch_val_mae = history.history['mae']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(range(0,len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
plt.plot(range(0,len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
plt.title('Evolution of loss on train & validation datasets over epochs')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(range(0,len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
plt.plot(range(0,len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2,label='Val MAE')
plt.title('Evolution of MAE on train & validation datasets over epochs')
plt.legend(loc='best')

plt.show()

#batch pequeño de tamaño 4 para evitar crash
results = model.evaluate(X_val, Y_val, batch_size=4)
print('Test loss:%3f'% (results[0]))
print('Test mae:%3f'% (results[1]))

predictTest()
print('END')
