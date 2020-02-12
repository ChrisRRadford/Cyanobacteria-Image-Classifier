#python3 DataTraining.py --image "/Volumes/EXTERNAL/ClassifierImageSets/Origional_2.png" --fileName "Training"

import cv2
import argparse
import numpy as np
import pandas as pd 
import wx

ap = argparse.ArgumentParser()
ap.add_argument("-1", "--Image", required=True, help="Image to be trained on")
ap.add_argument("-2", "--SaveName", required=True, help="Name of file to be saved")
args = vars(ap.parse_args())
image = cv2.imread(args["Image"])
fileName = (args["SaveName"])
currClass = -1
MasterList = np.empty((0,6), dtype = int)
count = 0
MasterClassCount = [0,0,0]

# onclick function
def click_and_crop(event, x,y,flags,param):
	global currClass, image, MasterList, MasterClassCount
	if event == cv2.EVENT_LBUTTONDBLCLK :
		if currClass not in range (1,4):
			print("Current Class not spceified Within Range (1,2,3)")
		else:
			refPt = [y,x]
			px = image[y,x]
			tempList = np.array([[refPt[1],refPt[0],px[0],px[1],px[2],currClass]])
			MasterList = np.append(MasterList,tempList,axis=0)
			if currClass == 1:
				cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(0,255,0),-1)
				MasterClassCount[0] +=1
			elif currClass == 2:
				cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(0,0,255),-1)
				MasterClassCount[1] +=1
			else:
				cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(122,122,122),-1)
				MasterClassCount[2] +=1
			print("Class count", MasterClassCount ,"To be added:", tempList, end="\r")

def main():
	app = wx.App(True) 
	width, height = wx.GetDisplaySize()
	del(app)

	global currClass,image, MasterList
	cv2.namedWindow('img',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('img', (int(width*0.9),int(height*0.9)))
	cv2.moveWindow("img", 20,20);
	cv2.setMouseCallback("img",click_and_crop)

	while(True):
		cv2.imshow("img",image)
		key = cv2.waitKey(1) & 0xFF
	 	
	 	# if the '1' key is pressed. Is Algae
		if key == ord("1"):
			print("Now selecting on Class 1 (Algae)")
			currClass = 1
		# if the '2' key is pressed. Isn't Algae
		if key == ord("2"):
			print("Now selecting on Class 2 (Non-Algae)")
			currClass = 2
		# if the '3' key is pressed. Is background
		if key == ord("3"):
			print("Now selecting on Class 3 (Background)")
			currClass = 3
		# if the 's' key is pressed. Save MasterList
		if key == ord("s"):
			print("Saving...")
			#SAVE HERE
			dataFrame = pd.DataFrame(data=MasterList, columns=["xCord","yCord","bBand","gBand","rBand","class"])
			path = fileName + ".csv"
			dataFrame.to_csv(path,index=False)
			print("Saved")
		# if the 'q' key is pressed. Quit
		if key == ord("q"):
			print("Exiting")
			break




main()