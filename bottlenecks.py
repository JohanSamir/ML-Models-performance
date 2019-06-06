#Codigo para producir bottlenecks
#importar librerias necesarias para el funcionamiento de este codigo
import os.path
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
 
 
#se define el nombre del nodo antes de la softmax para la inception  v3. El ":0" #quiere decir que vamos a recibir la salida de nodo pool3 de la inception v3,
#esto para poder crear los archivos bottlenecks, los cuales son unos archivos txt #que contienen 2048 valores numericos que son los datos de salida del nodo #pool3.
#Un bottleneck es creado para cada imagen que es procesada por la red, al #terminar con todo el dataset de imagenes, se tiene una cantidad de bottlenecks #igual al numero de imagenes para entrenamiento,
#este conjunto de bottlenecks seran el set de entrenamiento para una capa #clasificadora posteriormente.
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#tamano del bottleneck, el numero de datos que cada archivo txt contiene
BOTTLENECK_TENSOR_SIZE = 2048
#las imagenes de entrada a la red son tamano 299x299x3, internamente se hace #el "resize" de cada imagen
 
#se define el nombre del decodificador de imagenes que utiliza el graph de la #inception V3, este programa solo soporta imagenes JPG
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
#es la operacion que convierte las imagenes de entrada a (1x299x299x3) antes de #entrada a la primera capa.
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
 
#nombres de las carpeta donde se encuentra el modelo de la inception v3, las #imagenes de entrenamiento y donde se guardaran los bottlenecks.
model_dir = 'inception'
image_dir = 'Patches'
bottleneck_dir = 'bottlenecks'
 
 
 
def create_bottleneck_file(sess, jpeg_data_tensor, bottleneck_tensor):
    #se crea un vector vacio para guardar los nombres de las imagenes.
    file_list = []
    #se obtienen todos los nombre de las imagenes en la carpeta train que tengan                   
    #extension .jpg
    #file_glob = os.path.join(image_dir, '*.jpg')
    file_glob = os.path.join(image_dir, '*.jpg')
    #se organizan en el vector file_list todos las direcciones de las imagenes 
    #encontrados dentro de la carpeta train
    file_list.extend(gfile.Glob(file_glob))
    
        #para cada imagen dentro del vector file_list
    for image in file_list:
        #se obtiene el nombre de cada imagen, dividiendo la direccion y tomando el 
        #string mas al derecha el cual representa el nombre del archivo,
        #por ejemplo train/image.1.jpg, al realizar la division solo queda el 
        #image.1.jpg
        image_name = str(image).split('/')[-1]
        #se crea el path para guardar el bottleneck de cada imagen procesada
        path = bottleneck_dir+'/'+image_name+'.txt'
        #se imprime que se esta creando cada bottleneck
        print('Creating bottleneck at ' + path)
        
        #con fastGfile se carga la imagen como un string codificado a utf8
        image_data = gfile.FastGFile(image, 'rb').read()
        
        #se llama el metodo run_bottleneck_on_image y se envia la sesion, la 
        #imagen cargada, el decodificador de las imagenes y el tensor de salida de la 
        #inception v3
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        #convierte los 2048 valores del bottleneck a string y los separa por comas ","
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        #se guarda el bottleneck como un archivo txt
        with open(path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
 
 
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
        #la inception v3 procesa cada imagen y se obtienen los 2048 valores de la 
        #salida de la capa pool3
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    
    bottleneck_values = np.squeeze(bottleneck_values)
    
    return bottleneck_values
 
 
 
 
def create_inception_graph():
        #se crea una sesion en la GPU para poder cargar el graph, esto es necesario 
        #para que las operaciones sea ejecutadas y tensores sean evaluados.
    with tf.Session() as sess:
         #se obtiene la direccion del graph de la inception v3
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        #con la herramienta fastGfile de tensorflow de carga el graph_def de la 
        #inception v3 como un String
        with gfile.FastGFile(model_filename, 'rb') as f:
             #se define la variable graph_def como un objeto GraphDef vacio
            graph_def = tf.GraphDef()
            #el modelo de la inception v3 es f cargado como string se analiza(Parse) 
            #dentro de la variable ya definida
            graph_def.ParseFromString(f.read())
            #del modelo de la inception v3 cargado como graph_def se extraen los 
	 #datos del tensor de salida de los bottlenecks, el tensor del decodificador 
	 #de las imagenes, y el tensor del cambio de tamano de las imagenes de 
	 #entrada
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
    #retorna el graph, el tensor de la salida de los bottlenecks, el tensor del 
    #decodificador de las imagenes, y el tensor del cambio de tamano de las 
    #imagenes de entrada
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor
 
 
 
def main():
    #se llama el metodo create_inception_graph, el cual carga el graph_def de la  
    #inception v3
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
        #Se crea una sesion como sess
    sess = tf.Session()
        #se envian los datos del tensor del decodificador de las imagenes, el tensor 
        #de la salida de los bottlenecks y la sesion anterior, para crear los archivos 
        #bottleneck
    create_bottleneck_file(sess, jpeg_data_tensor, bottleneck_tensor)
 
if __name__ == '__main__':
    main()
