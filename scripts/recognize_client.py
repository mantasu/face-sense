import rospy
from face_sense.srv import FRClientGoal
from face_sense.utils import load_dict, verify_path

class RecognizeClient:
    def __init__(self, config_path, is_relative=True):
        self.config = load_dict(verify_path(config_path, is_relative))
        self.service_name = self.config["recognize"]["node"]["service_name"]
        rospy.wait_for_service(self.service_name)

    def send_client_goal(self, order_id, order_argument=None):
        try:
            proxy = rospy.ServiceProxy(self.service_name, FRClientGoal)
            response = proxy(order_id, order_argument)
            return response
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    config_path = "config.json"
    is_relative = True

    client = RecognizeClient(config_path, is_relative)

    while True:
        x = input("Enter order ID\n")
        x = int(x)

        if x == 0:
            y = input("Enter order argument\n")
        else:
            y = None
        
        response = client.send_client_goal(x, y)
        print(response)

        if x == 5:
            break