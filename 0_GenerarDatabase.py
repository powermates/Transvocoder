# -*- coding: utf-8 -*-
"""

La utilidad de crear esta base de datos es poder hacer consultas complejas fáciles de entender sobre el CSV.

@author: Powermates
"""
import sqlite3
import pandas as pd
import os

basepath='c:/VOICE_es' #ruta en la que está el contenido extraido del zip de Mozilla Commonvoice. CAMBIAR
dataframe_name='validated.tsv' #nombre del CSV tabular de metadatos de las muestras validadas de Mozilla Commonvoice. NO CAMBIAR

#Carga csv con ayuda de Pandas
mozilla_metadata = pd.read_csv(os.path.join(basepath, dataframe_name), sep='\t')

#creación o apertura de la base de datos
conn = sqlite3.connect("database_val.db") #(si no existe la crea)

#función Pandas para volcar la tabla del CSV en la bdd
mozilla_metadata.to_sql('validated', conn)


#Para probar distintas consultas se recomienda la creación de índices en la columna sencence y gender 
sql = 'CREATE INDEX "sentence_gender_idx" ON "validated" ("sentence","gender")'
cur = conn.cursor()
cur.execute(sql)
conn.commit()

