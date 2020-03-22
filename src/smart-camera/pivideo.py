# Owner: Joseph Wong
# Last updated: 3/21/20
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2

class PiVideo:
	def __init__(self, resolution=(640,480), framerate=5, **kwargs):
		"""
        Description: Start camera thread
        Parameters: class object, resolution, frame rate, option configuration
        Return: None
        """

		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate

		# Optional configuration
		for (arg, value) in kwargs.items():
			setattr(self.camera, arg, value)

		# Initialize stream
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)

		self.frame = None
		self.stopped = False

	def start(self):
		"""
        Description: Start camera thread
        Parameters: class object
        Return: self
        """

		t = Thread(target=self.update, args=())
		t.start()
		return self

	def update(self):
		"""
        Description: Loop for camera stream
        Parameters: class object
        Return: None
        """

		# Loop until thread ends
		for f in self.stream:
			# get frame from stream
			self.frame = f.array
			self.rawCapture.truncate(0)

			# Stop stream if stopped set to true
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return

	def read(self):
		"""
        Description: Returns current frame
        Parameters: class object
        Return: frame
        """

		return self.frame

	def stop(self):
		"""
        Description: Stop camera thread
        Parameters: class object
        Return: None
        """

		self.stopped = True
