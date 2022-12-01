import cv2

def Centroid(box, frame):
  x = int(box[0] + (box[2]/2))
  y = int(box[1] + (box[3]/2))
  cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
  return (x, y)