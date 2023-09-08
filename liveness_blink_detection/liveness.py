# # import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import exiftool
import sys
import time
import imutils

import logging

from library.util.image import imread
from library.face_detector import FaceDetector
from library.face_antspoofing import SpoofingDetector

logging.basicConfig(filename='/home/harshit/citybank_backup/sept2023/data/CityBank/CitybankDocs/mobiid/livenessdetection/loglog.log',level=logging.DEBUG)


class BlinkEyeDetection:

	shape_predictor="/home/harshit/citybank_backup/sept2023/data/CityBank/CitybankDocs/mobiid/livenessdetection/face_landmarks.dat"
	facial_landmarks_index = {'mouth': (48, 68),
		'inner_mouth': (60, 68),
		'right_eyebrow': (17, 22),
		'left_eyebrow': (22, 27),
		'right_eye': (36, 42),
		'left_eye': (42, 48),
		'nose': (27, 36),
		'jaw': (0, 17)}

	def __init__(self, predictor_model = None):
		if(predictor_model is not None):
			self.shape_predictor = predictor_model

		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.shape_predictor)
		self.face_detector = FaceDetector("/home/harshit/citybank_backup/sept2023/data/CityBank/CitybankDocs/mobiid/livenessdetection/data/pretrained/retina_face.pth.tar")
		self.face_antispoofing = SpoofingDetector("/home/harshit/citybank_backup/sept2023/data/CityBank/CitybankDocs/mobiid/livenessdetection/data/pretrained/fasnet_v1se_v2.pth.tar")

	def eye_aspect_ratio(self,eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear

	def detect_eye_blink(self, video_path, store_image_path = "temp.jpg", STD_THRESH = 0.015):
		"""
		define two constants, one for the eye aspect ratio to indicate
		blink and then a second constant for the number of consecutive
		frames the eye must be below the threshold
		"""
		# initialize the frame counters and the total number of blinks
		#logging.info(video_path)
		logging.info(store_image_path)

		video_path = video_path.strip()
		store_image_path = store_image_path.strip()

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(lStart, lEnd) = self.facial_landmarks_index["left_eye"]
		(rStart, rEnd) = self.facial_landmarks_index["right_eye"]

		# start the video stream thread
		vid = cv2.VideoCapture(video_path)
		eyeaspectratiolist = []
		# loop over frames from the video stream
		video_open_counter = 0
		while not vid.isOpened() and video_open_counter <3:
			vid = cv2.VideoCapture(video_path)
			video_open_counter+=1
			#logging.info("Try " + str(video_open_counter))
			time.sleep(2)

		rotation_value = 0
		with exiftool.ExifTool() as et:
			metadata = et.get_metadata(video_path)
			try:
				rotation_value = metadata['Composite:Rotation']
			except Exception as e:
				rotation_value = 0
			#logging.info("Rotation value of file :: " + str(rotation_value))

		max_ear = 0
		max_frame = None
		#logging.info('video start')
		fake_count = 0
		face_found_count = 0
		framecount = 0
		while True:
			flag, frame = vid.read()
			if not flag:
				break
			framecount+=1
			if(rotation_value == 180):
				frame = cv2.rotate(frame,cv2.ROTATE_180)
			elif(rotation_value == 90):
				frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
			elif(rotation_value == 270):
				frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)

			# frame = imutils.resize(frame, height=800)
			frame = imutils.resize(frame, height=480)
			rects = self.detector(frame, 0)
			# If does not find face or more than 1 face
			if (len(rects) != 1):
				continue
			rect = rects[0]
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = self.predictor(frame, rect)
			shape = np.asarray(list(map(lambda p: (p.x, p.y), shape.parts())))

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			eyeaspectratiolist.append(ear)
			
			if ear > max_ear:
				max_frame = frame

			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			faces = self.face_detector(image)
			if(len(faces) == 0):
				continue
			face_found_count+=1
			preds = self.face_antispoofing([box for box, _, _ in faces], image)
			
			if(preds[0][0] == False):
				fake_count+=1

		if not max_frame is None:
			cv2.imwrite(store_image_path,max_frame)
		vid.release()
		#logging.info('video end')
		outvalue = np.std(eyeaspectratiolist)
		blink = False

		if(outvalue>= STD_THRESH):
			blink = True
		logging.info(blink)
		logging.info(face_found_count/framecount)
		logging.info(fake_count/face_found_count)
		if( (face_found_count/framecount) >= 0.50 and (fake_count/face_found_count)<0.1 and blink):
			return True

		return False


if __name__=='__main__':
	print(f'This is args : {sys.argv}')
	try:
		if(len(sys.argv)<2):
			exit()
		video_path = sys.argv[1]
		if(len(sys.argv)>2):
			output_path = sys.argv[2]
		else:
			output_path = "temp.jpg"
		logging.info('start')
		blinkdetectionobj =BlinkEyeDetection()
		#logging.info('object create')
		result = blinkdetectionobj.detect_eye_blink(video_path,output_path)
		if(result):
			print("true")
		else:
			print("false")
		logging.info('end')
	except Exception as e:
		print(e)
		logging.error(e)
