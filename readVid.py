import cv2
from sys import exit, argv
import time


def loadDevice():	
	cap = cv2.VideoCapture('http://127.0.0.1:4747/mjpegfeed?640x480')
	if(cap.isOpened() == False):
		print("Error accessing video!")
		exit(1)
	else:
		print("Successfully streaming camera.")
		print("FPS:", cap.get(cv2.CAP_PROP_FPS))

	return cap


if __name__ == '__main__':
	cap = loadDevice()

	folder = "images3/mummy/"
	imgId = 0

	start = time.time()
	while(True):
		ret, frame = cap.read()

		if(ret == True):
			# frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.rotate(frame, cv2.ROTATE_180)
			cv2.imshow('frame', frame)
			
			# imgName = folder + "img{:06d}.jpg".format(imgId)
			# cv2.imwrite(imgName, frame)
			# imgId += 1
			# print("Time: {}".format(time.time()-start))

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		else:
			break

	end = time.time()
	print("Total time for recording: {}".format(end-start))

	cap.release()
	cv2.destroyAllWindows()
