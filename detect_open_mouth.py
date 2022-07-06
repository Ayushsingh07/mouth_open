from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def mouth_aspect_ratio(mouth):
	
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57


	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	
	mar = (A + B) / (2.0 * C)

	
	return mar


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

MOUTH_AR_THRESH = 0.79


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(mStart, mEnd) = (49, 68)

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

frame_width = 640
frame_height = 360

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	rects = detector(gray, 0)

	
	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		
		mouth = shape[mStart:mEnd]

		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		
		mouthHull = cv2.convexHull(mouth)
		
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		

     
		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
	
	out.write(frame)
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
