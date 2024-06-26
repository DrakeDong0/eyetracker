import cv2
import os
import time
import openai
from dotenv import load_dotenv
import numpy
import mediapipe

mp_face_mesh = mediapipe.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
      print(results.multi_face_landmarks)
    cv2.imshow('img', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
