import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torchvision
from sys import argv, exit
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import os
import PIL
import time


def loadTrainTest(dataDir, miniBatchSize=4, validSize=0.2):
	trainTransform = transforms.Compose([
		# transforms.Resize([224, 224]),
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	
	testTransform = transforms.Compose([
		# [transforms.Resize([224, 224]),
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224), 
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	trainData = datasets.ImageFolder(dataDir, transform=trainTransform)
	testData = datasets.ImageFolder(dataDir, transform=testTransform)
	
	classNames = trainData.classes

	totalData = len(trainData)
	indices = list(range(totalData))
	np.random.shuffle(indices)
	split = int(np.floor(validSize * totalData))
	trainIdx, testIdx = indices[split:], indices[:split]
	trainSampler = SubsetRandomSampler(trainIdx) 
	testSampler = SubsetRandomSampler(testIdx)
	trainloader = torch.utils.data.DataLoader(trainData, sampler=trainSampler, batch_size=miniBatchSize)
	testloader = torch.utils.data.DataLoader(testData, sampler=testSampler, batch_size=miniBatchSize)

	print("Class Names are: {}".format(classNames))

	return trainloader, testloader, classNames


def imshow(inp, title=None):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	
	if title is not None:
		plt.title(title)
	
	plt.imshow(inp)
	plt.show()


def showSampleData(trainloader, classNames):
	inputs, labels = next(iter(trainloader))
	out = torchvision.utils.make_grid(inputs)
	imshow(out, title=[classNames[x] for x in labels])


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


def getOptimParam(model):
	criterion = nn.NLLLoss()
	# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	expLrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	return criterion, optimizer, expLrScheduler


def evaluate(model, device, testloader, criterion):
	testLoss = 0
	testAcc = 0

	model.eval()
	with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model.forward(inputs)
			batchLoss = criterion(logps, labels)
			testLoss += batchLoss.item()

			ps = torch.exp(logps)
			topP, topClass = ps.topk(1, dim=1)
			equals = topClass == labels.view(*topClass.shape)
			testAcc += torch.mean(equals.type(torch.FloatTensor)).item()

	testAcc = 100 * testAcc/len(testloader)
	testLoss = testLoss/len(testloader)

	return testLoss, testAcc


def train(model, device, optimizer, criterion, trainloader, testloader, weightName, epochs):
	miniBatch = 0
	runningLoss = 0.0
	printMiniBatch = 100
	bestAcc = 0.0
	bestWts = copy.deepcopy(model.state_dict())
	trainLosses, testLosses = [], []

	startTime = time.time()
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			miniBatch += 1
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			runningLoss += loss.item()

			if(miniBatch%printMiniBatch == 0):
				testLoss, testAcc = evaluate(model, device, testloader, criterion)

				trainLoss = runningLoss/printMiniBatch 
				trainLosses.append(trainLoss)
				testLosses.append(testLoss)

				print("Epoch: {}/{}, Minibatch: {}/{}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}"
					.format(
					epoch+1,
					epochs,
					miniBatch,
					epochs*len(trainloader),
					trainLoss, 
					testLoss,
					testAcc))
				
				runningLoss = 0.0
				
				model.train()

				if(testAcc > bestAcc):
					bestAcc = testAcc
					bestWts = copy.deepcopy(model.state_dict())
					torch.save(bestWts, weightName)

		scheduler.step()
	endTime = time.time()

	print("Training completed in {:.4f} seconds".format(endTime - startTime))
	
	model.load_state_dict(bestWts)
	# torch.save(bestWts, weightName)

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

	return index, torch.exp(predict).data.cpu().numpy()


def evalImages(dataDir, model, device, classNames):
	classFolder = classNames[2]
	imgFiles = os.listdir(dataDir+classFolder)

	correctCount = 0

	for i, imgFile in enumerate(imgFiles):
		try:
			img = PIL.Image.open(os.path.join(dataDir, classFolder, imgFile))
		except IOError:
			continue

		index, probs = predictImage(img, model, device)
		# print("{}. Image belongs to class: {} | Probabilities: {}".format(
		# 	i, classNames[index], probs))

		# plt.imshow(np.asarray(img))
		# plt.show()

		if(classNames[index] == classFolder):
			correctCount += 1

	print("Accuracy for {} class is: {:.4f} | Correct Prediction: {} | Total Images: {} ".format(
		classFolder, correctCount*100/len(imgFiles),
		correctCount,
		len(imgFiles)))


if __name__ == '__main__':
	dataDir, argTrain = argv[1], int(argv[2])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	weightName = "person3.pth"
	miniBatchSize = 32
	epochs = 10
	numClasses = 4

	trainloader, testloader, classNames = loadTrainTest(dataDir, miniBatchSize)

	showSampleData(trainloader, classNames)

	model = getModel2(device, numClasses)
	totalParams = sum(p.numel() for p in model.parameters())
	trainParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total parameters: {} | Trainable parameters: {}".format(totalParams, trainParams))

	criterion, optimizer, scheduler = getOptimParam(model)

	if(argTrain == 1):
		model = train(model, device, optimizer, criterion, trainloader, testloader, weightName, epochs)

	model.load_state_dict(torch.load(weightName))
	testLoss, testAcc = evaluate(model, device, testloader, criterion)
	print("Final Accuracy: {:.4f} and Loss: {:.4f}".format(testAcc, testLoss))

	evalImages(dataDir, model, device, classNames)
