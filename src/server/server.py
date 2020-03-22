# Owner: Joseph Wong
# Last updated: 3/21/20
import os
import imutils
import socket
import cv2
import pickle
import numpy as np
import globals as g

from package_detection import package_recognition
from facial_recognition import facial_recognition

from threading import Thread
import tensorflow as tf
from tensorflow.python.util import deprecation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DEBUG = 0 # change to 1 for debugging print statements

#Server config
HOST = socket.gethostbyname(socket.gethostname())
print("[INFO] Host: %s" % HOST)
SERVER_PORT = 8000
BUFFER_SIZE = 4096
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#function to receive and append to candidate queues
def receive_frames():
    """
    Description: Main running thread, receives frames from smart camera
                 Saves frames to local directory
                 Appends frames to global lists
    Parameters: None
    Return: None
    """

    print("[INFO] Receive frames thread started")
    global most_recent_frame

    trigger_frame = 0

    try:
        while True:
            conn, addr = SERVER_SOCKET.accept()

            data = b''
            while True:
                block = conn.recv(4096)
                if not block: break
                data += block

            conn.close()

            final_image = pickle.loads(data,encoding='bytes') # store data recieved from smart camera as a numpy image

            #Save image into folder
            cv2.imwrite('trigger_frames/frame_' + str(trigger_frame).zfill(4) + '.jpg', final_image)

            if DEBUG:
                print('[INFO] Saved trigger frame')

            # global lists of image to check
            g.facial_recognition_candidates.append(final_image)
            g.facial_recognition_candidates.append(trigger_frame)
            g.package_candidates.append(final_image)
            g.package_candidates.append(trigger_frame)

            most_recent_frame = final_image

            trigger_frame+=1 # update frame number

    except KeyboardInterrupt:
        if not conn is None:
            conn.close()
        if not SERVER_SOCKET is None:
            SERVER_SOCKET.close()
        print("Exiting")

def server_setup():
    """
    Description: Configuration for socket connection
    Parameters: None
    Return: None
    """

    SERVER_SOCKET.bind((HOST, SERVER_PORT))
    SERVER_SOCKET.listen(5)
    receive_frames()
    SERVER_SOCKET.close()
    print('[INFO] Client disconnected')

#remove previous images in folders
def clear_frame_directories():
    """
    Description: Clears directories that store images in local directories
    Parameters: None
    Return: None
    """

    for frame in os.listdir('trigger_frames'):
        file_path = os.path.join('trigger_frames', frame)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    """
    Description: Main function clears directories that store images
                 Start threads to run functions
    Parameters: None
    Return: None
    """

    clear_frame_directories()

    # start all threads for functions
    Thread(target=server_setup).start()
    Thread(target=facial_recognition).start()
    Thread(target=package_recognition).start()
