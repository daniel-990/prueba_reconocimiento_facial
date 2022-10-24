import cv2 as cv
import os
#import imutils

modelo="FotosBrol" #Nombre de la carpeta donde se guardan las fotos (Datos de entrada).
ruta1="./Data" #Ruta donde se creara la carpeta.
rutacompleta= ruta1 + "/"+ modelo
if not os.path.exists(rutacompleta):
	os.makedirs(rutacompleta)

camara=cv.VideoCapture(0)
ruido=cv.CascadeClassifier("./opencv/data/haarcascades/haarcascade_frontalface_default.xml")
id=0

while True:
	respuesta, captura=camara.read()
	if respuesta==False:
		break

    #captura=imutils.resize(captura,width=640)

	grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)

    #idcaptura=captura.copy()

	cara=ruido.detectMultiScale(grises,1.3,5)
	for(x,y,e1,e2) in cara:
		cv.rectangle(captura, (x,y), (x+e1, y+e2), (0, 255, 0), 2)
		#Esta líneas de código generan 350 foto del personaje a reconocer y las etiqueta con un número y extensión .JPG
		rostrocapturado=captura[y:y+e2, x:x+e1]
		rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
		cv.imwrite(rutacompleta+"/imagen_{}.jpg".format(id), rostrocapturado)
		id=id+1

	cv.imshow("Imagen Rostro", captura)

	if id==50: #Este if detiene la creación de imágenes al llegar a la imagen 350.JPG y cierra el programa.
		break
camara.release()
cv.destroyAllWindows()