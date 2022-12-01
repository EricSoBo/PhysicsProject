import cv2

def selectROIfromFrame(fr):
  # box (x, y, lar, alt)
  box = cv2.selectROI("SELECT ROI", fr, fromCenter=False, showCrosshair=False)
  print(box)
  return box

def ObjectDetectionAI(fr):
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []
    file_name = 'labels.txt'
    with open(file_name, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

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