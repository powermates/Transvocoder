# -*- coding: utf-8 -*-
"""
***

Ejecuta la SQL y la carga en Pandas dataframe

Se recomienda la creación de índices en la columna sencence y client_id sobre la base de datos 

Selecciona la consulta más idónea

Convierte el audio a escala mel y lo guarda a disco en formato numpy

***

@author: Powermates
"""
import sqlite3
import pandas as pd
import librosa
import os
import numpy as np


#convertimos a mels, convertimos a escala logarítmica de -80..+80db, y normalizamos con rango -1 a 1
def preprocess(wave):
    array = librosa.feature.melspectrogram(y=wave, n_mels=n_mels) #salida float32
    array = librosa.power_to_db(array)
    array = array/80
    return array

#padding de ceros a la derecha del audio hasta llegar a 100000
#el efecto real es esto 
def rpad(wave,n_pad=100000):
    dif=n_pad-len(wave)
    wave=np.concatenate((wave, np.zeros(dif,dtype=np.float32)), axis=None)
    return wave


#creación o apertura de la base de datos
conn = sqlite3.connect("database_val.db") #(si no existe la crea)


#Comentar y dejar solo una de las tres consultas de debajo. 
#El objetivo de estas consultas es obtener en un dataframe el nombre de los archivos.mp3 de hombres y mujeres


#La primera contiene unos 7000 pares de registros y tarda un segundo por cada uno...
#La segunda unos 1000 pares
#La última(dfMozillaFAA) contiene menos de 200 registros con lo que es la más rápida, apenas unos minutos.


#1
#Esta consulta une a todas las frases iguales siempre que estén pronunciadas por un hombre y una mujer 
#y se asegura que el usuario sea distinto 
#sql_stringMaleFemale = "select * from validated v1, validated v2 where v1.sentence=v2.sentence and v1.client_id<>v2.client_id and v1.gender='male' and v2.gender='female' order by v1.sentence"
#dfMozilla = pd.read_sql(sql_stringMaleFemale, conn)
#print(dfMozilla)

#2
#Adicional a la anterior pero ademas asemeja el acento de las muestras
#sql_stringMaleFemaleAccent = "select * from validated v1, validated v2 where v1.sentence=v2.sentence and v1.client_id<>v2.client_id and v1.gender='male' and v2.gender='female' and v1.accent=v2.accent order by v1.sentence"
#dfMozillaFA = pd.read_sql(sql_stringMaleFemaleAccent, conn)
#print(dfMozillaFA)

#3
#Adicional a la anterior pero además une rangos de edad
sql_stringMaleFemaleAccentAge = "select * from validated v1, validated v2 where v1.sentence=v2.sentence and v1.client_id<>v2.client_id and v1.gender='male' and v2.gender='female' and v1.accent=v2.accent and v1.age=v2.age order by v1.sentence"
dfMozillaFAA = pd.read_sql(sql_stringMaleFemaleAccentAge, conn)
print(dfMozillaFAA)


basepath='c:/VOICE_es' #ruta en la que está el contenido extraido del zip de Mozilla Commonvoice. CAMBIAR

n_mels=128 #número de mels, 128 es suficiente para voz, pero podemos escoger de 32 a 512
tiempo=196 #número de muestras de los mel, 196 es lo que ocupan 100000 samples

X_train=[]
Y_train=[]

vueltas=0

""" *Generación de datos*
    El siguiente bloque hace lo siguiente con la consulta del dataframe seleccionado:
    - Carga el audio con Librosa (hace falta librerías externas como FFMPEG si son mp3 en lugar de wav) 
    - Baja el sampleo a 11025hz por motivos de eficiencia
    - Descarta muestras extrañas excesivamente largas
    - En cualquier caso agregamos un padding de ceros a la derecha del audio hasta llegar a 100000
    - Car

"""

for fila in dfMozillaFAA.itertuples():
    vueltas+=1
    if vueltas>7000: #podemos limitar el número de muestras, o no. CAMBIAR O LIMITAR PARA PROBAR
        break

    voice1=os.path.join(basepath + '/clips', fila[3])
    voice2=os.path.join(basepath + '/clips', fila[12])

    #carga la onda
    v1,sr1=librosa.load(voice1, sr=11025)
    v2,sr2=librosa.load(voice2, sr=11025)
    
    # Si la longitud del audio mayor a 100000 samples descarta ambas, 
    #debido a que el corpus MCV tiene muestras con fallos o muy largas
    if len(v1)<100000 and len(v2)<100000:
        #mostramos en el terminal las muestras que se están guardando y su longitud en segundos
        # segundos = longitud sample / sample rate
        print(vueltas, voice1, len(v1)/sr1 , sr1, ' vs ' + voice2, len(v2)/sr2,sr2, sep='\t')
        
        #agrega padding a la derecha hasta completar 100000 máximo
        v1=rpad(v1)
        v2=rpad(v2)
        
        #convertimos a mels, convertimos a escala logarítmica de 80db, y normalizamos con rango -1 a 1
        v1=preprocess(v1)
        v2=preprocess(v2)

        #añadimos los arrays resultantes a la lista de X y de Y
        X_train.append(v1)
        Y_train.append(v2)

#cambiamos el array para que contenga 4 dimensiones para las convolucionales 2D: batch, dim1, dim2, canales
X_train = np.reshape(X_train, (len(X_train), n_mels, tiempo, 1))
Y_train = np.reshape(Y_train, (len(Y_train), n_mels, tiempo, 1))

# convertir lista a numpy array
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)


#Salvamos el array a disco para su uso posterior
np.save('XAA.npy', X_train)
np.save('YAA.npy', Y_train)



