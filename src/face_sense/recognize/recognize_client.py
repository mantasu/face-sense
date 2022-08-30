import rospy

from threading import Thread, Event

from face_sense.srv import FRClientGoal
from face_sense.utils import load_dict, verify_path

class RecognizeClient:
    def __init__(self, config_path, is_relative=True):
        # Load the config dictionary, extract service name and worker
        self.config = load_dict(verify_path(config_path, is_relative))
        self.service_name = self.config["recognize"]["node"]["service_name"]
        self.worker_type = self.config["recognize"]["node"]["worker_type"]

        # Maps worker types to methods
        self.WORKER_MAP = {
            "command_line": self.command_line_worker
        }

        # Assign the thread target based on the worker type
        self.thread_target = self.WORKER_MAP[self.worker_type]

        # Initialize thread-related
        self.event_stop = Event()
        self.thread = None

        # Wait till the server is initiated
        rospy.wait_for_service(self.service_name)

    def send_client_goal(self, order_id, order_argument=None):
        try:
            # Create the service proxy and send the order ID + its arg 
            proxy = rospy.ServiceProxy(self.service_name, FRClientGoal)
            response = proxy(order_id, order_argument)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        
        return response
    
    def command_line_worker(self):
        while True:
            if self.event_stop.is_set():
                # Clear stop event, end 
                self.event_stop.clear()
                break
            
            # Obtain x and y values from shell
            x = int(input("Enter order ID\n"))
            y = input("Enter order argument\n") if x == 0 else None

            if x == 5:
                # Exit if 5
                self.stop()
            
            # Execute the goal and generate response
            response = self.send_client_goal(x, y)
            rospy.loginfo(response)
    
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            # Init thread and start running based on target
            self.thread = Thread(target=self.thread_target)
            self.thread.start()
    
    def stop(self):
        # Stop event to true
        self.event_stop.set()
