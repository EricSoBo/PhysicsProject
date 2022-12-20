import cv2
import numpy as np
from IdentificarCentroide import Centroid

def selectROIfromFrame(fr):
  # box (x, y, lar, alt)
  box = cv2.selectROI("SELECT ROI", fr, fromCenter=False, showCrosshair=False)
  print(box)
  return box

def Tracking(Video):
	frame_rate = 1 # em milisegundos quanto demora um frame, é apenas para visulização...

	cap = cv2.VideoCapture(Video) # Abre o vídeo
	if (not cap.isOpened()): 
	  print("Erro ao abrir o arquivo")

	#img_object = cv2.imread('objects\Ball.png') # Se for imagem
	# Salto com vara: (353, 410, 30, 60)
	# Basquete: (557, 140, 20, 28)
	ret, first_frame = cap.read()

	first_frame = cv2.resize(first_frame, (800, 600)) # Se for frame

	#box = selectROIfromFrame(img_object)
	box = selectROIfromFrame(first_frame) # Tracking
	#box = (557, 140, 20, 28) # Define manualmente a área a ser analisada

	tracker = cv2.TrackerCSRT_create()
	tracker.init(first_frame, box)

	#tracker.init(img_object, box)

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

	return X, Y, Time

def MultiTracking(Objects, Video):
	frame_rate = 1 # em milisegundos quanto demora um frame, é apenas para visulização...

	cap = cv2.VideoCapture(Video) # Abre o vídeo
	if (not cap.isOpened()): 
	  print("Erro ao abrir o arquivo")

	#img_object = cv2.imread('objects\Ball.png') # Se for imagem
	# Salto com vara: (353, 410, 30, 60)
	# Basquete: (557, 140, 20, 28)
	ret, first_frame = cap.read()

	first_frame = cv2.resize(first_frame, (800, 600)) # Se for frame

	trackers = cv2.legacy.MultiTracker_create()
	for i in range(Objects):
		box = selectROIfromFrame(first_frame)
		tracker = cv2.legacy.TrackerCSRT_create()
		trackers.add(tracker, first_frame, box)

	X, Y = dict(), dict()
	Time = np.array([], dtype=np.float64)
	PosCentroid = [0]*Objects
	
	for i in range(Objects):
		X['Object'+str(i)], Y['Object'+str(i)] = np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)

	while (cap.isOpened()):

		ret, frame = cap.read()

		if not ret:
		   break

		frame = cv2.resize(frame, (800, 600))
		ok, boxes = trackers.update(frame)
		if ok:
			for i, box in enumerate(boxes):
				pt1 = (int(box[0]), int(box[1]))
				pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
				cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2, 1)
				PosCentroid[i] = Centroid(box, frame)

		else:
			cv2.putText(frame, "Tracking falhou", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

		if cap.get(cv2.CAP_PROP_POS_MSEC) != 0:
			for i in range(Objects):
				X['Object'+str(i)] = np.append(X['Object'+str(i)], PosCentroid[i][0])
				Y['Object'+str(i)] = np.append(Y['Object'+str(i)], PosCentroid[i][1])
			Time = np.append(Time, cap.get(cv2.CAP_PROP_POS_MSEC))
		cv2.imshow('Tracking',frame)

		if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	
	return X, Y, Time

def ObjectDetectionAI(Video):

	config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
	frozen_model = 'frozen_inference_graph.pb'

	classLabels = []
	file_name = 'labels.txt'
	with open(file_name, 'rt') as fpt:
		classLabels = fpt.read().rstrip('\n').split('\n')

	model = cv2.dnn_DetectionModel(frozen_model, config_file)
	model.setInputSize(320, 320)
	model.setInputScale(1.0/127.5)
	model.setInputMean((127.5, 127.5, 127.5))
	model.setInputSwapRB(True)

	frame_rate = 1 # em milisegundos quanto demora um frame, é apenas para visulização...

	cap = cv2.VideoCapture('samples\Collision_1.mp4') # Abre o vídeo
	if (not cap.isOpened()): 
		print("Erro ao abrir o arquivo")
	while (cap.isOpened()):

		ret, fr = cap.read()

		if not ret:
			break

		ClassIndex, confidence, bbox = model.detect(fr, confThreshold=0.55)
		print(ClassIndex)

		font_scale = 3
		font = cv2.FONT_HERSHEY_PLAIN
		if (len(ClassIndex) != 0):
			for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
				cv2.rectangle(fr, boxes, (255, 0, 0), 2)
				cv2.putText(fr, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    
		final = cv2.resize(fr, (800, 600))
		cv2.imshow('Object Detection', final)

		if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
			break