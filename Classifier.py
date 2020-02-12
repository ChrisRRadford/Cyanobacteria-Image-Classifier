import argparse
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA as sklearnPCA #need
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import cluster
from sklearn.cluster import KMeans # KMeans clustering 
import cv2 	#need
import lxml	#need
import collections #need
from tqdm import tqdm #need
import wx				#need

# Blur detection function used to deretmine if the overall iamge is sharp enough to be used
def blurDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
	filtered = cv2.Laplacian(gray,cv2.CV_64F);
	variance = filtered.var()
	return variance

# Image enhancement phase of identified algorithm 
def imageEnhancement(image):
	#seperate out the bands
	b,g,r = cv2.split(image)
	rBandMean = np.mean(r)
	height,width = image.shape[:2]
	#stated variable with algorithm
	c = 1
	#traverse through pixels and adjsut as needed
	print("Iterating through each pixel and enhancing...")
	#rows
	for i in tqdm(range(height)):
		#cols
		for j in range(width):

			pixel = r[i,j]
			
			if(pixel > rBandMean):
				r[i,j] = pixel + min(255, pixel + ((255 - rBandMean) * c ))
			else:
				r[i,j] = max(0,pixel - ((rBandMean - 1)*c))
	#merge bands with updated r band
	image = cv2.merge((b,g,r))
	return image
# Classification utilising PCA to reduce PC bands and Kmeans clustering
def classification(enhancedImaged,trainingData,initialK,img,width,height):
	b,g,r = cv2.split(enhancedImaged)
	# Pandas dataset
	dataSet = pd.DataFrame({'bBand':b.flat[:],'gBand':g.flat[:],'rBand':r.flat[:]})

	#print(dataSet.head())
	# Standardize the data
	X = dataSet.values
	X_std = StandardScaler().fit_transform(X) #converts data from unit8 to float64

	#Calculating Eigenvectors and eigenvalues of Covariance matrix
	meanVec = np.mean(X_std, axis=0)
	covarianceMatx = np.cov(X_std.T)
	eigVals, eigVecs = np.linalg.eig(covarianceMatx)
	#print(dataSet.shape)

	# Create a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [ (np.abs(eigVals[i]),eigVecs[:,i]) for i in range(len(eigVals))]
	# Sort from high to low
	eig_pairs.sort(key = lambda x: x[0], reverse= True)

	# Determine how many PC going to choose for new feature subspace via
	# the explained variance measure which is calculated from eigen vals
	# The explained variance tells us how much information (variance) can 
	# be attributed to each of the principal components
	tot = sum(eigVals)
	var_exp = [(i / tot)*100 for i in sorted(eigVals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)

	#plot shows that all data is contained within first to PC bands
	#convert 3 dimension space to 2 dimensional space therefore getting a 2x3 matrix W
	matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
	                      eig_pairs[1][1].reshape(3,1)))

	#use matrix W to transfrom samples onto new subspace via equation Y=X*W
	Y = X_std.dot(matrix_w)

	# Run PCA
	print("Executing Principal Component Analysis on data. Reducing to 2 bands.")
	sklearn_pca = sklearnPCA(n_components = 2)
	x_3d = sklearn_pca.fit_transform(X_std)

	#Variables for while loop

	KmeansClusterInitial = initialK
	tempImage = img.copy()

	#-------Will continue to re-runing increase the number of classes used base on user input--------
	while(True):
		print("Executing kmeans clustering on PCA adjusted data with",KmeansClusterInitial, " clusters")
		#Set a 4 KMeans clustering
		kmeans = KMeans(n_clusters = KmeansClusterInitial, n_jobs = -2,random_state=42)

		#Compute cluster centers and predict cluster indices
		classAssignment = kmeans.fit_predict(x_3d)
		# Create a temp dataframe from our PCA projection data "x_3d"
		df = pd.DataFrame(x_3d)
		row,col = enhancedImaged.shape[:2]
		count = 0
		print("Mapping Clusters data to image...")
		for i in tqdm(range(row)):
			for j in range(col):
				if classAssignment[count]==0:
					tempImage[i,j] = (0,255,0)
				elif classAssignment[count]==1:
					tempImage[i,j] = (255,255,255)
				elif classAssignment[count]==2:
					tempImage[i,j] = (0,0,255)
				elif classAssignment[count]==3:
					tempImage[i,j] = (125,125,0)
				elif classAssignment[count]==4:
					tempImage[i,j] = (0,125,125)
				elif classAssignment[count]==5:
					tempImage[i,j] = (125,0,125)
				elif classAssignment[count]==6:
					tempImage[i,j] = (255,255,0)
				elif classAssignment[count]==7:
					tempImage[i,j] = (255,0,255)
				elif classAssignment[count]==8:
					tempImage[i,j] = (0,255,255)
				elif classAssignment[count]==9:
					tempImage[i,j] = (125,125,125)  
				count+= 1

		#Display everything for user to utilise
		template = cv2.imread("KmeansClusterTemplate.png")
		cv2.namedWindow('Classes',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Classes', (int(width*0.45),int(height*0.45)))
		cv2.moveWindow("Classes", 20,20);
		cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Original', (int(width*0.45),int(height*0.45)))
		cv2.moveWindow("Original", int(width*0.45)+50,20);
		cv2.namedWindow("Colour Classification",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Colour Classification", (int(width*0.45),int(height*0.45)))
		cv2.moveWindow("Colour Classification", int(width*0.45)-int((width*0.45)/2),int(height*0.45)+50);
		
		
		#------User Input------
		while(True):
			cv2.imshow('Classes',tempImage)
			cv2.imshow('Original',img)
			cv2.imshow("Colour Classification",template)
			print("When ready, press ENTER to proceed (You will no longer able to manipulate images)")
			cv2.waitKey(0)
			algae = [int(x) for x in input("Please select classes deemed to be Algae: ").split()]
			nonAlgae = [int(x) for x in input("Please select classes deemed to be non Algae: ").split()]
			background = [int(x) for x in input("Please select classes deemed to be the background: ").split()]

			#------Check for Valid Class Inputs------
			#merge lists
			fullList = algae + nonAlgae + background
			#create checkList of number of classes
			classList = list(range(0,KmeansClusterInitial))
			#compare they are the same
			compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
			check = compare(classList,fullList)
			if check:
				print("Valid class selections")
				break
			#List isn't valid. Find problem to help user
			else:
				#Check if there are duplicate values
				if len(fullList) > len(set(fullList)):
					print("Not unique. A class was referenced twice")
				else:
					print("A class stated is out of the given range of valid classes: 0 -", KmeansClusterInitial-1) 

		cv2.destroyAllWindows()

		#Reclassify based on users input
		print("Reclassifying to create mask...")
		classified = img.copy()
		count = 0
		for i in tqdm(range(row)):
			for j in range(col):
				kClass = classAssignment[count]
				#If class is stated as algae
				if kClass in algae:
					classified[i,j] = (255,255,255)
					classAssignment[count] = 1
				#If class is stated as non algae
				elif kClass in nonAlgae:
					classified[i,j] = (0,0,0)
					classAssignment[count] = 2
				#If class is stated as backgrounf
				else:
					classified[i,j] = (222,222,222)
					classAssignment[count] = 3
				count+= 1
		#Add updated classAssignment too new values
		dataSet['label'] = classAssignment
		#dataSet.to_csv("/Users/chrisradford/Documents/School/Masters/RA/Classifier/Python/Training.csv",index=False)
			
		#-------Display Mask and convolution Matrix to determine is satisfied---------
		#Convolution Matrix
		if trainingData.empty:
			print("No Training Data provided, therefore not running confusion accraucy assessment")
		else:
			print("Running Confusion Matrix Accuracy Assessment using provided training data....")

			#Create a new Dataframe for confusion matrix only using coordinate of Training Data points
			row,col = img.shape[:2]
			pred = []
			test = []
			for index,row in trainingData.iterrows():
				#print("----------")
				c = row["xCord"]
				r = row["yCord"]
				indexVal = r*col +c
				pred.append(row['class'])
				test.append(dataSet.iloc[indexVal]['label'])

			print(confusion_matrix(test,pred))

		#Display Final Mask
		cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Mask', (int(width*0.9),int(height*0.9)))
		cv2.moveWindow("Mask", 20,20);
		#Dtermine if user is satisfied with results
		while(True):
			cv2.imshow("Mask",classified)
			print("When ready, press ENTER to proceed (You will no longer able to manipulate images)")
			cv2.waitKey(0)
			#Determine if you are satisfied with the results of the classifcaiton to
			response = input("Are you satisified with these results (y/n)? ")
			
			if(response == 'y'):
				cv2.destroyAllWindows()
				return classified
			elif(response == 'n'):
				KmeansClusterInitial += 1
				cv2.destroyAllWindows()
				print("Re-running classification with an increased number of classes: (",KmeansClusterInitial,")")
				break
			else:
				print("Invalid response...")
				continue

		

	#If leaving while loop you were unabale to perfrom classification
	Print("Classification could not be performed on this image at this time")
	exit()

def colorCorrection(img):
	#Color Correction
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))
	hsv[:,:,1] = greenMask 
	final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return final

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-1", "--Image", required=True, help="Image to be trained on")
	ap.add_argument("-2", "--TrainingData", required=False, help="Training Data")
	ap.add_argument("-3", "--Threshold", required=False, help="Blur threshold")
	ap.add_argument("-4", "--SaveName", required=True, help="Name of file to be saved")
	ap.add_argument("-5", "--ClassNumber", required=False, help="Number of initial classes to use for KMeans Clustering")
	args = vars(ap.parse_args())
	#grab variables from agrparse
	img = cv2.imread(args["Image"],cv2.IMREAD_COLOR)
	fileName = (args["SaveName"])
	#Check if inital kmeans clusters is set
	if args['ClassNumber'] is not None:
		givenKmeans = int(args['ClassNumber'])
	else:
		print("No Class number stated for KMeans Clustering. Default set at 4.")
		givenKmeans = 4

	#Check if training Data is set
	if args['TrainingData'] is not None:
		df = pd.read_csv(args["TrainingData"])
	else:
		df = pd.DataFrame()

	#Check if a blur threshold has been given
	if args["Threshold"] is not None:
		blurThreshold = int(args["Threshold"])
		print("---Running Blur Detection---")
		variance = blurDetection(img)
		#print(variance)
		if variance < blurThreshold:
			print("too blurry")
			exit()
		print("Image Sharpness Satisfactory...")
		variance = None
	#begin image enhancement check
	print("---Beginning Image Enhancement---")
	enhancedImaged = imageEnhancement(img)
	#enhancedImagedSmall = cv2.resize(enhancedImaged,(0,0), fx=0.5, fy=0.5)
	print("---Beginning Classification---")

	tempImage=enhancedImaged.copy()
	app = wx.App(False)
	width, height = wx.GetDisplaySize()	#display window options
	print(width,height)
	del(app)

	mask = classification(enhancedImaged,df,givenKmeans,img,width,height)
	print("Executing final mask overlay")
	print()
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	imageRemapped = cv2.bitwise_and(img,img,mask=mask)
	finalResult = colorCorrection(imageRemapped)
	cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result', (int(width*0.9),int(height*0.9)))
	cv2.moveWindow("Result", 20,20);
	while(True):
		cv2.imshow("Result",finalResult)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	path = fileName + ".png"
	cv2.imwrite(path,finalResult)


	

main()




