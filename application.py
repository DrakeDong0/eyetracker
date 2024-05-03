import cv2
import os
import time
import numpy
import mediapipe
import math
import pyautogui

mp_face_mesh = mediapipe.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469,470,471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]
TOTAL_BLINKS = 0
CEF_COUNTER = 0
CLOSED_EYES_FRAME = 1
last_blink_time = None
blink_count = 0

def euclidean_distance(point1, point2):
  x1, y1, =point1.ravel()
  x2, y2, =point2.ravel()
  distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
  return distance

def iris_position(iris_center, right_point, left_point):
  center_to_right_dist = euclidean_distance(iris_center, right_point)
  total_distance = euclidean_distance(right_point, left_point)
  ratio = center_to_right_dist/total_distance
  iris_position = ""
  if ratio <= 0.42:
    iris_position = "right"
  elif ratio > 0.42 and ratio <= 0.57:
    iris_position="center"
  else: 
    iris_position = "left"     
  return iris_position, ratio

def blinkRatioRightEye(landmarks, right_indices):
    epsilon = 1e-6
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)

    reRatio = rhDistance / (rvDistance + epsilon)
    return reRatio

def blinkRatioLeftEye(landmarks, left_indices):
    epsilon = 1e-6
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    lhDistance = euclidean_distance(lh_right, lh_left)
    lvDistance = euclidean_distance(lv_top, lv_bottom)

    leRatio = lhDistance / (lvDistance + epsilon)
    return leRatio
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  while True:
    #get current cursor location
    x, y = pyautogui.position()
    print("x: " + str(x) + "|" + "Y: " + str(y))
    
    
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
      mesh_points = numpy.array([numpy.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
      print(mesh_points.shape)
      cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
      cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
      (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
      (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
      center_left = numpy.array([l_cx, l_cy], dtype=numpy.int32)
      center_right = numpy.array([r_cx, r_cy], dtype=numpy.int32)
      cv2.circle(frame, center_left, int(l_radius), (0,0,255), 1, cv2.LINE_AA)
      cv2.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)
      
      cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1, cv2.LINE_AA)
      cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1, cv2.LINE_AA)
      iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
      print(iris_pos)
      cv2.putText(frame, f"Iris position: {iris_pos} {ratio:.2f}", (30,30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
      Leftratio = blinkRatioLeftEye(mesh_points, LEFT_EYE)
      RightRatio = blinkRatioRightEye(mesh_points, RIGHT_EYE)
      
      
      
      if(iris_pos == "right"):
        pyautogui.moveTo( (x+200), y, 0)
      if(iris_pos == "left"):
        pyautogui.moveTo((x-200), y, 0)
        
      if (Leftratio > 5.3) and (RightRatio < 4):
        pyautogui.click(button="right")
      else:
        if Leftratio > 5.3 and RightRatio > 5.3:  # Assuming that a ratio > 5.3 means the eye is closed
          CEF_COUNTER += 1
        else:
          if CEF_COUNTER > CLOSED_EYES_FRAME:
            current_time = time.time()
            if last_blink_time is not None:
              if (current_time - last_blink_time) < 0.3:
                blink_count += 1
                if blink_count >= 2:
                  blink_count = 0
                  pyautogui.click()
                else:
                  blink_count = 1
              else:
                blink_count = 1
            last_blink_time = current_time

          CEF_COUNTER = 0
      cv2.imshow('img', frame)
      key = cv2.waitKey(1)
      if key == ord('q'):
          break
cap.release()
cv2.destroyAllWindows()
