# Owner: Joseph Wong
# Last updated: 3/21/20
import face_recognition
import pickle
import os
import glob

from sklearn import svm, neighbors, tree, ensemble

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
datasets = ['person1','person2','person3'] # corresponds to folder names in train_dir directory

# Loop through each person in the training directory
for person in datasets:
    path_to_data = './train_dir/' + person + '/'
    filelist = sorted(glob.glob(path_to_data + '*.jpg'))

    # Loop through each training image for the current person
    for filename in filelist:
        print("[INFO] Training on: " + filename.split('/')[-1])

        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(filename)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains none or more than one face, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(filename + " contains none or more than one face")
            exit()
        else:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

# User SVM classifier and save as .pickle
clf = svm.SVC(gamma='scale') #Support vector classification
clf.fit(encodings,names)
f = open("svc.pickle", "wb")
pickle.dump(clf,f,-1)
f.close()