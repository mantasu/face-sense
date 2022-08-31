import rospy

from threading import Thread, Event

from face_sense.learn.general import Trainer
from face_sense.learn.specific import FaceDataset
from face_sense.recognize.components.core_recognizer import CoreRecognizer

class RecognitionGoalHandler:
    """Handles recognition goals

    This class has the main method ~RecognitionGoaHandler.handle_order
    which handles a recognition goal based on its order ID and an
    optional order argument. Currently the following IDs are accepted:
        * `0` - generates a set of images and stores them in the
            identity folder. This order also takes an additional string
            argument which should be the name of the identity to capture
            the photos of.
        * `1` - generates a face embeddings file which contains face
            embeddings from the identity folder. It is based on the
            face analysis app which is based on the specified face
            detection/analysis model.
        * `2` - trains the face classifier to recognize face embedding
            given the embeddings dataset.
        * `3` - recognizes face from the given image once. Given the
            face app specification and the classifier specification, it
            extracts information about the face, such as name, bounding
            box, age etc. Multiple faces are supported.
        * `4` - recognizes the face in the renewing images continuously.
             Given the face app specification and the classifier
             specification, a thread is created where every specified
             interval of time information about the face is extracted.
             Multiple faces are supported.
        * `5` - exits the recognition process. It is only an indicator
            for upper classes that there should be no more goals
            handled.
    """
    def __init__(self, config):
        """Initializes the recognition goal handler

        Simply takes a configuration dictionary and initializes certain
        attributes. It does not handle any publishers/messages.

        Args:
            config (dict): The recognizer configuration dictionary,
                i.e., parsed dict from `config.json`
        """
        # Initialize the core recognizer (detection + recognition)
        self.recognizer = CoreRecognizer(config["inference"])
        
        # Main attributes
        self.end = None
        self.frame = None
        self.identities = None

        # Thread attributes
        self.thread = None
        self.event_stop = Event()
        self.start_time = rospy.get_time()
        self.interval = config["node"]["process_interval"]

        # Only learn config is needed
        self.config = config["learn"]

    def _prepare_for_order(self, order_id):
        """Prepares to execute the desired goal (based on order ID)

        This method verifies that the appropriate objects have been
        initialized for the desired goal. For example, if the client
        wants to perform face recognition but the model is not yet
        created, execution will be aborted with a warning message.
        The following checks are performed:
            1. `order_id` = 2 (train model): it is checked whether the
                face embeddings file (training data) is present.
            2. `order_id` = 3 or `order_id` = 4 (recognize face): it is
                checked whether the face embeddings file (inference
                data) is present, whether Face App (for face extraction)
                is created based on the specified model, and whether the
                Classification Model (for face recognition) is created
                based on the specified `.pth` file.
            3. `order_id` != 4 (not continuous recognition): it is
                checked whether the goal is _NOT_ to recognize
                continuously and if that's true, the thread, if exists,
                for continuous recognition is stopped.

        Args:
            order_id (int): The type of goal to prepare to

        Returns:
            str: An empty string if preparation is successful and an
                error/warning message if preparation failed
        """
        if order_id == 2 or order_id == 3 or order_id == 4:
            if self.recognizer.embeddings is None:
                # Check if embeddings file is available
                self.recognizer.init_data(verbose=True)

                if self.recognizer.embeddings is None:
                    return "Cannot load embeddings file."
        
        if order_id == 3 or order_id == 4:
            if self.recognizer.app is None or self.recognizer.model is None:
                # Check if face app and classifier are available
                self.recognizer.init_models(verbose=True)

                if self.recognizer.model is None:
                    return "Cannot load recognizer model."

        if order_id != 4 and self.thread is not None and self.thread.is_alive():
            # Stop thread for continuous recognition if it is running
            self.event_stop.set()
            self.thread.join()
        
        return ""
    
    def handle_order(self, order_id, order_argument=None):
        """Executes the specified order

        Given an order ID, this method does some preparation and, if it
        is successful, executes the order based on its ID (some orders
        may require an additional argument) and returns a response of
        how did the execution go.

        Args:
            order_id (int): The ID of the desired goal
            order_argument (str): Additional argument in case the order
                execution demands it

        Returns:
            str: A response on how successful the order went
        """
        # Do some preparation before executing order
        response = self._prepare_for_order(order_id)

        if response != "":
            # If prep failed
            return response

        if order_id == 0:
            response = self.generate_identities(order_argument)
        elif order_id == 1:
            response = self.generate_embeddings()
        elif order_id == 2:
            response = self.train_model()
        elif order_id == 3:
            response = self.recognize_once()
        elif order_id == 4:
            response = self.recognize_continuous()
        elif order_id == 5:
            response = self.exit()
        
        return response
    
    def generate_identities(self, name, num_faces=10):
        return "Functionality is not yet available!"

    def generate_embeddings(self):
        """Generates the face embeddings file

        Given the path to the identity photos in the `config.json` file,
        this method generates a face embeddings file in a specified
        directory.

        Returns:
            str: A success message showing the path of the embeddings
        """
        # Extract the required arguments
        app = self.config["face_analysis"]
        photo_dir = self.config["data"]["photo_dir"]
        embed_dir = self.config["data"]["embed_dir"]
        is_relative = self.config["data"]["is_relative"]
        
        # Call the static method of the FaceDataset object to gen embeds
        path = FaceDataset.gen_embeds(app, photo_dir, embed_dir, is_relative)
        
        return f"Embeddings file generated in {path}"
    
    def train_model(self, embed_name=None):
        """Trains the face classifier

        Uses the specified embeddings file to initialize the training
        dataset and sets up a desired classifier which is trained on the
        created dataset.

        Note: it is possible to send the goal to generate embeddings
            first, then send the goal to train the model on those
            embeddings. Just make sure to set `embed_name` to "newest"
            in `config.json` which ensures the most recently modified
            embedding is selected.

        Returns:
            str: A success message indicating the training finished
        """
        # Extract the required args and init dataset
        embed_dir = self.config["data"]["embed_dir"]
        embed_name = self.config["data"]["embed_name"]
        is_relative = self.config["data"]["is_relative"]
        dataset = FaceDataset(embed_dir, embed_name, is_relative)
        
        # Initialize the trainer and start training
        trainer = Trainer(self.config, dataset)
        trainer.run()

        return f"Training finished successfully"
    
    def recognize_once(self):
        """Recognizes the face in a current frame once

        Performs a single inference to recognize the faces in the image
        (current frame) and returns the name(-s) of the recognized
        person (people).

        Returns:
            str: A success message showing the first name of the
                recognized person
        """
        # Name placeholder
        name = "Unknown"

        if self.frame is not None:
            # Call the core recognizer object to process the image
            self.identities = self.recognizer.process(self.frame)

            if len(self.identities["names"]) != 0:
                # Get the name if someone's recognized
                name = self.identities["names"][0]

        return f"Recognized {name}"

    def recognize_continuous(self):
        """Recognizes faces continuously

        Every `self.interval` seconds, `self.frame` (which should be
        regularly updated/reset) is processed to detect and recognize
        all the faces in that frame.

        Note: this goal does not end automatically, the thread ends
            after next order is required to handle.

        Returns:
            str: An indication message that continuous recognition
                started successfully
        """
        def thread_fn():
            while True:
                if self.event_stop.is_set():
                    # If recognition should stop
                    self.event_stop.clear()
                    break
                
                if rospy.get_time() - self.start_time < self.interval:
                    # Don't process if minimum interval not reached
                    continue
                else:
                    # Otherwise reset the interval timer
                    self.start_time = rospy.get_time()
                
                if self.frame is not None:
                    # Given the current stream frame, recognize faces
                    self.identities = self.recognizer.process(self.frame)
        
        # Start continuous recognition thread
        self.thread = Thread(target=thread_fn)
        self.thread.start()

        return "Started recognizing continuously."

    def exit(self):
        """Sets exit indicator to True

        This simply sets `self.end` to True to indicate that there will
        be no more order requests

        Returns:
            str: A message indicating the system is exiting
        """
        self.end = True

        return "Exiting..."
    
    def set_frame(self, frame):
        """Updates current frame

        Sets a new provided image as a current frame (`self.frame`)

        Args:
            frame (numpy.ndarray): The frame to set as current
        """
        self.frame = frame.copy()

    def get_identities(self):
        """Gets the recognized identities

        Gets the currently recognized identities. This will be `None` if
        no recognition has yet occurred. If multiple requests are
        issued, same value will be returned if the recognition happened
        only once and a changing value based on `self.interval` if the
        recognition thread is running (self.thread).

        Returns:
            dict: A dictionary with keys indicating features, such as
                "names", "genders", and values indicating identities,
                i.e, lists where each entry is a feature that the key
                specifies
        """
        return self.identities    