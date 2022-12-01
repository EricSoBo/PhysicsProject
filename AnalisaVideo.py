import cv2
import numpy as np
from DetectarObjetos import ObjectDetectionAI, selectROIfromFrame
from IdentificarCentroide import Centroid
from AnalisaDados import HarmonicOscAmort, ProjectileMotion

frame_rate = 1 # em milisegundos quanto demora um frame, é apenas para visulização...

cap = cv2.VideoCapture('samples\AmortHarmoOsc.mp4') # Abre o vídeo
if (not cap.isOpened()): 
  print("Erro ao abrir o arquivo")

#img_object = cv2.imread('objects\Ball.png') # Se for imagem
# Salto com vara: (353, 410, 30, 60)
# Basquete: (557, 140, 20, 28)
ret, first_frame = cap.read()

first_frame = cv2.resize(first_frame, (800, 600)) # Se for frame

#box = selectROIfromFrame(img_object)
box = selectROIfromFrame(first_frame)
#box = (557, 140, 20, 28) # Define manualmente a área a ser analisada

tracker = cv2.TrackerCSRT_create()
#tracker.init(img_object, box)
tracker.init(first_frame, box)

X, Y = np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)
Time = np.array([], dtype=np.float64)

while (cap.isOpened()):

  ret, frame = cap.read()

  if not ret:
   break

  frame = cv2.resize(frame, (800, 600))
  ok, box = tracker.update(frame)
  if ok:
    pt1 = (box[0], box[1])
    pt2 = ((box[0] + box[2]), (box[1] + box[3]))
    cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2, 1)
    PosCentroid = Centroid(box, frame)
  else:
    cv2.putText(frame, "Tracking falhou", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

  if cap.get(cv2.CAP_PROP_POS_MSEC) != 0:
    X = np.append(X, PosCentroid[0])
    Y = np.append(Y, PosCentroid[1])
    Time = np.append(Time, cap.get(cv2.CAP_PROP_POS_MSEC))
  cv2.imshow('Tracking',frame)

  if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

HarmonicOscAmort(X, Y, Time)
#ProjectileMotion(X, Y, Time)