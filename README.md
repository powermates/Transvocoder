# Transgender Vocoder

El presente proyecto trata de demostrar la capacidad de las redes convolucionales en la transformación de voces.

# Manual de usuario
Requisitos:
-	Librerias: Tensorflow 2, Librosa, SQLite
-	Programas: FFMPEG/SDL o compatibles instalada en el sistema operativo, para poder cargar archivos mp3.

Preparación:
-	Descargar el dataset de Mozilla Commonvoice y guardarlo en una carpeta local. Es muy grande, son 13 gigabytes y 100.000 archivos con lo que se aconseja un disco duro rápido.
-	Eso crea una carpeta llamada VOICE_es con todo el contenido.
-	Posteriormente hay que editar el primer fichero 0_GenerarDatabase.py para corregir dicha ruta 
-	Elegir en 1_Preprocesado.py la SQL y dataframe que queremos utilizar para generar los datos. Estas consultas están comentadas en el código.

Ejecución: 
- 0_GenerarDatabase.py
- 1_Preprocesado.py
- 2_Entrenamiento_Test_RCED.py

Para la repetición solo habría que ejecutar #2 ya que los anteriores ejecutables guardan a disco los arrays numpy con las ondas preprocesadas y normalizadas listas para usar.
Los correspondientes a las 7000 muestras ocupan 600 megabytes cada array, con lo que no se han subido a Github, pero sí se han subido unos pequeños (XAA.npy YAA.npy) para poder probar sin el corpus.
En el caso de 2.py hay que editarlo y cambiar el nombre del archivo de entrada en la función predictTest() male.mp3 o usar uno propio de menos de 10 segundos de duración.

Se ha usado una GPU nVidia GTX 980TI de 6 gigas de memoria y un tamaño de batch de 5, ya que valores cercanos a 10 hacen fallar el modelo por out of memory.

Se incluyen en el repositorio los arrays numpy para X e Y de tamaño reducido.

NOTA: Actualmente el resultado no es favorable pero se considera de utilidad didáctica.
