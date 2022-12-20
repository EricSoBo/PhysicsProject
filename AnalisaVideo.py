import cv2
import numpy as np
import matplotlib.pyplot as plt
from DetectarObjetos import ObjectDetectionAI, MultiTracking, Tracking
from AnalisaDados import HarmonicOscAmort, ProjectileMotion, PendCollision, MinQuadrados

X, Y, Time = MultiTracking(2, 'samples\Collision_1.mp4')

PendCollision(X, Y, Time, 2)