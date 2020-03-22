# Owner: Joseph Wong
# Last updated: 3/21/20
import os
import cv2
import socket
import pickle
import imutils
import time
import live_stream
import numpy as np

from camera import VideoCamera
from cStringIO import StringIO
from threading import Thread

video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically

#Set server IP address
SERVER_IP = '172.20.10.2'

send_frames_flag = 1 # change to 1 if you want frames to be sent to server

def send_frame_to_server(frame):
    """
    Description: Sends frame to server via socket
    Parameters: frame
    Return: None
    """

    if send_frames_flag:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((SERVER_IP, 8000))

        serialized_frame = pickle.dumps(frame, protocol=2)
        client.sendall(serialized_frame)
        client.close()

def clear_frame_directories():
    """
    Description: Clears directories that store images in local directories
    Parameters: None
    Return: None
    """

    frames_folders = ['all_frames','trigger_frames']
    for folder_name in frames_folders:
        for frame in os.listdir(folder_name):
            file_path = os.path.join(folder_name, frame)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

def detect_motion():
    """
    Description: Detects motion between two frames
    Parameters: None
    Return: None
    """

    global video_camera

    frame_number = 0 #frame counter for file name

    while(True):
        # get frame
        frame = video_camera.get_frame()
        live_stream.live_frames.append(frame)

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # get grayscale image
        blurred_gray = cv2.GaussianBlur(current_gray, (21, 21), 0) # get blurred grayscale image

        # not the first frame (frame_number != 0)
        if not frame_number:
            cv2.imwrite('all_frames/frame_' + str(frame_number) + '.jpg', current_gray)
            frame_number+=1
            previous_frame = blurred_gray
            continue

        # get the absolute difference between frames
        frameDelta = cv2.absdiff(previous_frame, blurred_gray)
        pixel_threshold = 20
        thresh = cv2.threshold(frameDelta, pixel_threshold, 255, cv2.THRESH_BINARY)[1] # set threshold array

        # get contours of thresholded image (get white blobs)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        motion = False
        contour_size = 700

        # only care about blobs bigger than 700 pixels
        # greater than 700 means motion detected
        for c in cnts:
            if cv2.contourArea(c) < contour_size:
                continue
            motion = True

        # send frame to server if motion was detected
        if motion == True:
            print "TRIGGER FRAME"
            # uncomment to save images locally
            #cv2.imwrite('trigger_frames/frame_' + str(frame_number) + '.jpg', current_gray)
            #cv2.imwrite('all_frames/frame_' + str(frame_number) + '.jpg', current_gray)

            send_frame_to_server(frame)
        else:
            print "NON-TRIGGER FRAME"
            #cv2.imwrite('all_frames/frame_' + str(frame_number) + '.jpg', current_gray)

        #always save every frame
        cv2.imwrite('all_frames/frame_' + str(frame_number) + '.jpg', frame)

        frame_number+=1 # update frame number
        previous_frame = blurred_gray # set current frame as next frame to be compared to

    del video_camera

if __name__ == '__main__':
    """
    Description: Main function clears directories that store images
                 Start threads to run functions
    Parameters: None
    Return: None
    """

    clear_frame_directories()

    print("[INFO] Connecting to server: " + SERVER_IP)

    Thread(target=detect_motion).start()
    time.sleep(3) # allow buffer time before live stream starts
    Thread(target=live_stream.start_live_stream).start()
