# Owner: Joseph Wong
# Last updated: 3/21/20
import pickle
import face_recognition
import globals as g

from threading import Thread
from send_sms import send_alert

f = open("../../models/svm.pickle", "rb") # model trained to recognize known faces
clf = pickle.load(f,encoding='latin1')
f.close()

known_faces = ['joe','mark'] # names of two known faces

Thread(target=send_alert).start() # start thread to receive alert to send SMS

DEBUG = 0

def facial_recognition():
    """
    Description: Classifies faces
    Parameters: None
    Return: None
    """

    print("[INFO] Facial recognition thread started")

    while(True):
        if g.facial_recognition_candidates: # loop through global lists in order if they ar enot empty
            image = g.facial_recognition_candidates[0]
            frame_number = g.facial_recognition_candidates[1]

            # Find all the faces in the frame
            face_locations = face_recognition.face_locations(image)
            no = len(face_locations)

            # found no faces, continue to next frame
            if no == 0:
                if DEBUG:
                    print("[INFO] No person detected in trigger frame " + str(frame_number))
                g.facial_alert_queue.append(frame_number)
                g.facial_alert_queue.append('no_person')
                del g.facial_recognition_candidates[:2]
                continue

            # loop through faces found
            for i in range(no):
                if i==(no-1):
                    image_enc = face_recognition.face_encodings(image)[i]
                    name = clf.predict([image_enc]) # predict name

                    if DEBUG:
                        print("[INFO] Detected: " + name[0] + " from trigger frame " + str(frame_number))

                    # handle prediction
                    if name[0] not in known_faces:
                        if DEBUG:
                            print("[INFO] Detected unknown person from trigger frame " + str(frame_number))
                        g.facial_alert_queue.append(frame_number)
                        g.facial_alert_queue.append('unknown_person')
                    else:
                        g.facial_alert_queue.append(frame_number)
                        g.facial_alert_queue.append('known_person')

            del g.facial_recognition_candidates[:2] # remove from global list
