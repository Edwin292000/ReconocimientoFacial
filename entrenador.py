# coding=utf-8

import cv2
import os
import numpy as np

ruta = 'D:/Users/edwin/Desktop/UNI/teoria de sistemas/deteccion de rostros/data' #ruta del proyecto
listado = os.listdir(ruta)
print('Lista de personas: ', listado)

labels = []
facesData = []
label = 0

for direct in listado:
	persona = ruta + '/' + direct
	print('Leyendo las imágenes')

	for fileName in os.listdir(persona):
		print('Rostros: ', direct + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(persona+'/'+fileName,0))
		imagen = cv2.imread(persona+'/'+fileName,0)
		cv2.imshow('imagen',imagen)
		cv2.waitKey(10)
	label = label + 1


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando")
face_recognizer.train(facesData, np.array(labels))


face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado")

