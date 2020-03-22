# Owner: Joseph Wong
# Last updated: 3/21/20
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Load all necessary models
detector = cv2.dnn.readNetFromCaffe("face_detection_model/deploy.prototxt", "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("face_detection_model/openface_nn4.small2.v1.t7")

# Get list of images paths
imagePaths = list(paths.list_images("train_dir")) # directory should include any images with a face

knownEmbeddings = []
knownNames = []

numFaces = 0

# Loop through the image paths
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Preprocess images
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)

	# Get grayscale and store as 3d array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.dstack([gray] * 3)

	(h, w) = image.shape[:2]

	# Get image blob
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# Open CV locates faces
	detector.setInput(imageBlob)
	detections = detector.forward()

	# Only care if face was found
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# Make sure face is actually a face
		# Add face embedding to list
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			numFaces += 1

# Store embeddings in dictionary and save as .pickle
print("[INFO] serializing {} encodings...".format(numFaces))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
