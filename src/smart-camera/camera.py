# Owner: Joseph Wong
# Last updated: 3/21/20
import cv2
import time
import numpy as np
from pivideo import PiVideo

class VideoCamera(object):
    """
    Description: Video camera class
    """

    def __init__(self, flip = True):
        """
        Description: Constructor method start camera
        Parameters: flip - bool to flip the frame or not, class object
        Return: None
        """

        self.vs = PiVideoStream().start()
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        """
        Description: Desctructor method stops camera
        Parameters: None
        Return: None
        """

        self.vs.stop()

    def get_frame(self):
        """
        Description: Get current frame from smart-camera
        Parameters: None
        Return: frame
        """

        frame = np.rot90(self.vs.read(),2)
        return frame
