import cv2
import rospy

from sensor_msgs.msg import CompressedImage, Image
from face_sense.msg import FaceInfo
from cv_bridge import CvBridge, CvBridgeError

class CameraViewer:
    def __init__(self, config, win_name="Camera Frames"):
        # Dims & win name
        camera_topic = config["camera_topic"]
        face_info_topic = config["face_info_topic"]
        image_type = CompressedImage if config["is_compressed"] else Image
        self.win_name = win_name

        # A communication bridge
        self.bridge = CvBridge()

        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)
        self.info_subscriber = rospy.Subscriber(
            face_info_topic, FaceInfo, self.info_callback, queue_size=1)
    
    def cam_callback(self, data):
        try:
            # Retrieve the compressed, image and resize it
            frame = self.bridge.compressed_imgmsg_to_cv2(data)
            # frame_scaled = cv2.resize(frame, (self.width, self.height))

            # Create a resizable window and load the frame/image
            cv2.imshow(self.win_name, frame)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)

        if cv2.waitKey(1) & 0xFF in [27, 113] or \
           cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
            # If `Q`|`ESC` key or window exit button clicked 
            rospy.signal_shutdown("Quit button clicked")
    
    def info_callback(self, data):
        pass