READ ME FILE

Installation Requirements:
The following are the command line commands to run in terminal/command Promt to get the reuired packages to run the following programs:

	Step 1 - Install Python 3 (just google and download appropriate version for your machine)

	Step 2 - Install pip within terminal/command prompt 
 				- "sudo easy_install pip"

 	(If you are told you do not have permissions to run these commands as "sudo" to beginning of command)
 	Step 3 - Install necessary packages
 				- "pip3 install pandas"
 				- "pip3 install sklearn"
 				- "pip3 install numpy"
 				- "pip3 install matplotlib"
 				- "pip3 install lxml"
 				- "pip3 install scipy"
 				- "pip3 install opencv-python"
 				- "pip3 install tqdm"
 				- "pip3 install wxPython"

 	Step 4 - Make sure all the modules improted correctly

 				- "python3"	(This should put you into a python shell in which you run the following code snippets)
 				- "import pandas"
 				- "import sklearn"
 				- "import numpy"
 				- "import matplotlib"
 				- "import lxml"
 				- "import scipy"
 				- "import cv2"
 				- "import tqdm"
 				- "import wx"

 				All of these should individual import and no errors should occur. 
 				Possible error "Module not found". if this is the case simply google how to download. Can also try "pip install.." instead of "pip3 install..."

-----------------------------------------------------------

DataTraining Program

The DataTraining Program is designed to allow the user to manually idenfity pixels containing algae and non algae that can be used for accuracy assessment of the main program.

	Prameters:
		--Image (Required)		The full pathname to the image you wish to use (Must be wrapped in "")	i.e --Image "/Users/chrisradford/Documents/School/Masters/RA/Classifier/ImageSets/Orig_12.png"


		--SaveName (Required)	The name of the file you wish to store the training data (Must be wrapped in "") i.e --SaveName "Kingson Pond 7 Training Data"
			NOTE: If you wish to save the training data in a different folder then where the program is stored you will need to enter in a complete file path once again encased with "". The image will be saved as a .cas (You do not need to put extension in name)

	Example Input command in terminal(Assuming you are in the same working directory that the program is stored)
	-------------------
	python3 DataTraining.py --Image "/Users/chrisradford/Documents/School/Masters/RA/Classifier/ImageSets/Orig_12.png" --SaveName "Orig_12_DataTraining"
	-------------------

	Output:
		CSV file: Provided you speciifed to save your results (See below) a .csv file willl be saved at the location you specified and with the name you also provided as an input parameter.

	Usage:
		Select Class:
			You can either hit '1','2',or '3' to specify what class you are wishing to idenfity. The class is initally set to (-1) ensuring you do not accidently click a pixel while trying to manipulate the image
				1 = Algae
				2 = Non-Algae
				3 = background
		Manipulate Image View:
			Reccomended to using a standard mouse. You can right click and drag to pan across the image and use the scroll whell to zoom in and out
		Classify a Pixel:
			Provided you have stated the class you are currently defining (1,2,3) you simply double-left-click on the pixel itself. Upon doing this a sqaure will appear confiming your input and display a color depended on the class (1 = green, 2 = red, 3 = gray)
		Save your Training Data:
			When you are ready to save your training data hit 's' and you work will be saved to the location you specified as an input. You can save multiple times and your previous save file will be over written
		Quit the Program:
			When you are ready to quit the program simply hit 'q'. Note your data will not save unless you saved your training data before quiting. 

	Tips:
		1. It is a good idea to have at least 30 pixels classified for each of the 3 classes (minimum 90)

	SOMETHING WENT WRONG:
		1. Error (-215) - the image path you provided was bad and no image was loaded in. Make sure you image path is in "".
		2. Cannot input classes or hit 'q' and 's' - You are not currently working with the mage. Simplly move mouse over the image and click once.
		3. Could not save file - Either a; You added the file format at the end which is not needed. b; The filepath you spcfied to save to wasn't valid 

-----------------------------------------------------------

Classifier Program

The Classifier program is the main program used to classify images for cyano bacteria.

	Prameters:
		--Image (Required)	The full pathname to the image you wish to use (Must be wrapped in "")	i.e --Iamge "/Users/chrisradford/Documents/School/Masters/RA/Classifier/ImageSets/Orig_12.png"

		--SaveName (Required)	The name of the file you wish to store the resulting image (Must be wrapped in "") i.e --SaveName "Kingson Pond 7" 
			NOTE: If you wish to save the training data in a different folder then where the program is stored you will need to enter in a complete file path once again encased with "". The image will be saved as a .png (You do not need to put extension in name)

		--Threshold (Not Required) This is a number that is sued to ensure that the image is of sufficient sharpness and quality to act as a quality control. If not set the program will run without doing the quality control check. i.e --Treshold 80

		--TrainingData (Not Required) This is a .csv file that you sohuld ahve previously created using the DataTraining program. IF not set the program will run without outputting a confusion matrix accuracy assessment. i.e --TrainingData "/Users/chrisradford/Documents/School/Masters/RA/Classifier/Python/Orig_11_training.csv"

		--ClassNumber (Not Required) This is a number that is used to determine how many different clusters to use for the image pixel cluster. The higher the number the more clusters that will be found within the image. If not specified the computer will run with an inital value set to 4 i.e --ClassNumber 
 
	Example Input command in terminal(Assuming you are in the same working directory that the program is stored)
	-------------------
	python3 Classifier.py --Image "/Users/chrisradford/Documents/School/Masters/RA/Classifier/ImageSets/Orig_11.png" --TrainingData "/Users/chrisradford/Documents/School/Masters/RA/Classifier/Python/Orig_11_training.csv" --Threshold 80 --SaveName "Pond8" --ClassNumber 6

	Or (minimum requirements):

	python3 Classifier.py --Image "/Users/chrisradford/Documents/School/Masters/RA/Classifier/ImageSets/Orig_11.png" --SaveName "Pond8"
	-------------------

	Output:
		Classified Image: The output is autoamtically saved based on the --SaveName parameter. It is an image where all pixels are black except for the ones that were classified as Algae.

	Usage:
		Select Classes:
			1.Upon running the inital program you will see loading bars and text to help you track the program of the classification. Eventually 3 image will appear. The first image (top left) is a picture of the image with all its pixels being classified. The second image (top right) is the origional image for your reference. The final image (bottom middle) is a classification colour char you will use to determine classes

			2.You can miipulate the image by clicking on them then panning around or using the scroll wheel to zoom in and out. When you are ready to identify the classes; While click on one of the image hit the ENTER key.

			3.In the termminal/cmd promt window you will then see the text "Please select classes deemed to be Algae:" Here you will enter a list of numbers (sperated by a space) that represent Algae within your classified image (top left).

			4.In the termminal/cmd promt window you will then see the text "Please select classes deemed to be non Algae:" Here you will enter a list of numbers (sperated by a space) that represent non-Algae within your classified image (top left).

			5.In the termminal/cmd promt window you will then see the text "Please select classes deemed to be background:" Here you will enter a list of numbers (sperated by a space) that represent the background within your classified image (top left).

			6.Upon completing step 5 the program will check that you have inputted the classes correctly and either proceed if you did or have you re-input your classidications if not correclty inputed. 

		Confirming your results:
			1. Upon completion of correctly identifying your classes the program will output a mask with white pixels denoting Algae and black not. Once again you can manipulate the image as previously described. When you are ready to identify the classes; While click on one of the image hit the ENTER key.

			2. In the termminal/cmd promt window you will then see the text "Are you satisified with these results (y/n)?" If you enter 'y' the program will finish. If you enter 'n' the program will re-run its classifcaiton of your image with an increased number of classes to identify (you will have to repeat the above steps)

		Quiting the program:
			3. Upon finishing the claissifcaiton you should see the final result displayed. Once again you can manipulate the image as previously described. When you are ready to quit hit the 'q' key. The image at this point will be saved to where you specified. 

	SOMETHING WENT WRONG:
		1. Error (-215) - the image path you provided was bad and no image was loaded in. Make sure you image path is in "".
		2. Cannot input classes or hit 'q' and 's' - You are not currently working with the mage. Simplly move mouse over the image and click once.
		3. Could not save file - Either a; You added the file format at the end which is not needed. b; The filepath you spcfied to save to wasn't valid 
		4. Error with classification input - a.check that you didn't put in a negative number b.check that you didn't put in a non number character
		 










