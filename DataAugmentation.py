# Importing the Libraries to augment the dataset

import os
import argparse
import cv2
from tqdm import tqdm 
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

# Loading the Dataset for for Original Images and Masked Images and sort them 
def loadData(path):
	 images = sorted(glob(os.path.join(path, "images/*")))     
	 masks = sorted(glob(os.path.join(path, "masks/*")))
	 return images, masks

def createDirectory(path):
	if not os.path.exists(path):
		os.makedirs(path)


# Augmenting the Data

def augmentData(images, masks, savePath, augment=True):

	# Setting Height and Width for the Final Augmented Masks and Images
	Height = 256
	Width = 256
	# Augmented Data would be of 256 X 256 size

	for x, y in tqdm(zip(images, masks), total=len(images)):
		
		name = x.split("/")[-1].split(".")
		
		# Extracting the names and extensions of the images and the masks
		imageName = name[0]
		imageExtension = name[1]

		name = y.split("/")[-1].split(".")
		maskName = name[0]
		maskExtension = name[1]

		# Reading Inages and Masks

		x = cv2.imread(x, cv2.IMREAD_COLOR)
		y = cv2.imread(y, cv2.IMREAD_COLOR)

		# Augmenting the Data
		# Applying the Transformations

		if augment == True:
			aug = CenterCrop(Height, Width, p=1.0)
			augmented = aug(image=x, mask=y)
			xCroppedImage = augmented["image"]
			yCroppedMask = augmented["mask"]

			aug = RandomRotate90(p=1.0)
			augmented = aug(image=x, mask=y)
			xRotatedImage = augmented['image']
			yRotatedMask = augmented['mask']

			aug = GridDistortion(p=1.0)
			augmented = aug(image=x, mask=y)
			xGridDistortionImage = augmented['image']
			yGridDistortedMask = augmented['mask']

			aug = HorizontalFlip(p=1.0)
			augmented = aug(image=x, mask=y)
			xHorizontalFlipImage = augmented['image']
			yHorizontalFlipMask = augmented['mask']

			aug = VerticalFlip(p=1.0)
			augmented = aug(image=x, mask=y)
			xVerticalFlipImage = augmented['image']
			yVerticalFlipMask = augmented['mask']

			saveImages = [x, xCroppedImage, xRotatedImage, xGridDistortionImage, xHorizontalFlipImage, xVerticalFlipImage]
			saveMasks =  [y, yCroppedMask, yRotatedMask, yGridDistortedMask, yHorizontalFlipMask, yVerticalFlipMask]

		else:
			saveImages = [x]
			saveMasks = [y]


		 # Saving the images and Masks in the specified Masks

		sequenceNumber = 0


		for i, m in zip(saveImages, saveMasks):

			i = cv2.resize(i, (Width, Height))
			m = cv2.resize(m, (Width, Height))

			if len(images) == 1:
				newImageName = f"{imageName}.{imageExtension}"
				newMaskName = f"{maskName}.{maskExtension}"

			else:
				newImageName = f"{imageName}_{sequenceNumber}.{imageExtension}"
				newMaskName = f"{maskName}_{sequenceNumber}.{maskExtension}"

			imagePath = os.path.join(savePath, "images", newImageName)
			maskPath = os.path.join(savePath, "masks", newMaskName)

			cv2.imwrite(imagePath, i)
			cv2.imwrite(maskPath, m)

			sequenceNumber += 1


if __name__ == "__main__":

# Loading original images and masks

	arg = argparse.ArgumentParser()
	arg.add_argument("--pathIn", help="Path to folder that contains Images and Masks")
	args = arg.parse_args()

	#path = "CVC-612/"
	path = args.pathIn
	images, masks = loadData(path)
	print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

	# Creating folders.
	createDirectory("newData/images")
	createDirectory("newData/masks")

	# Applying data augmentation.
	augmentData(images, masks, "newData", augment=True)

	""" Loading augmented images and masks. """
	images, masks = loadData("newData/")
	print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")
