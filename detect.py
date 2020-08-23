import cv2
from sys import exit, argv

from torchvision import transforms, models
from torch import nn
import torch
from PIL import Image
import numpy as np


def getModel(device, numClasses):
	model = models.resnet50(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False

	model.fc = nn.Sequential(nn.Linear(2048, 512), 
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(512, numClasses),
		nn.LogSoftmax(dim=1))

	model.to(device)

	return model


def getModel2(device, numClasses):
	model = models.resnet18(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False

	model.fc = nn.Sequential(nn.Linear(512, 256), 
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(256, numClasses),
		nn.LogSoftmax(dim=1))

	model.to(device)

	return model


def getModel3(device, numClasses):
	model = models.resnet18(pretrained=True)

	# for param in model.parameters():
	# 	param.requires_grad = False

	model.fc = nn.Sequential(nn.Linear(512, numClasses), 
		nn.LogSoftmax(dim=1))

	model.to(device)

	return model


def predictImage(img, model, device):
	testTransform = transforms.Compose([
		# transforms.Resize([224, 224]), 
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	model.eval()
	with torch.no_grad():
		imgTensor = testTransform(img)
		imgTensor = imgTensor.unsqueeze_(0)
		imgTensor = imgTensor.to(device)	
		predict = model(imgTensor)
		index = predict.data.cpu().numpy().argmax()

	return index, torch.exp(predict).data.cpu().numpy().squeeze()


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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	weightName = "person3.pth"
	classNames = ['himanshi', 'mummy', 'papa', 'udit']

	model = getModel2(device, len(classNames))
	model.load_state_dict(torch.load(weightName))
	print("Model loaded Successfully.")

	bottomLeft, font, scale, color, lineType = (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 

	i = 0
	while(True):
		ret, frame = cap.read()

		if(ret == True):
			frame = cv2.rotate(frame, cv2.ROTATE_180)
			
			frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			imgPil = Image.fromarray(frame2)
			index, probs = predictImage(imgPil, model, device)
			
			text = "Label: {} | Probability: {:.2f}".format(classNames[index], np.max(probs))

			cv2.putText(frame, text, bottomLeft, font, scale, color, lineType)
			cv2.imshow('frame', frame)

			i += 1

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		else:
			break

	cap.release()
	cv2.destroyAllWindows()