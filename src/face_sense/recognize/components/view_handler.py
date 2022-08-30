import cv2

class RecognitionViewHandler:
    """A class that draws boxes, names etc on raw images with faces

    This class contains utility methods to draw certain items on an
    image with recognized faces. Specifically, it can draw bounding
    boxes, face landmarks, name with confidence score and bio
    information - gender and age. The main method to use is
    ~RecognitionViewHandler.draw_on_frame.

    Attributes:
        BOX_STYLE (tuple): Specifies the style for the bounding box of
            the face. It is a tuple containing a tuple for RGB color
            specification and a float number specifying the thickness
        MARKS_STYLE (tuple): Specifies the style for the face landmarks.
            It is a tuple containing a list of tuples specifying RGB
            colors for landmarks [left, right, middle] and a float value
            specifying the marks thickness
        NAME_STYLE (tuple): Specifies the style for the text containing
            the identity name and confidence score. It is a tuple
            containing cv2 enumerator for font face, a float value
            specifying the text scale, a tuple specifying RGB color and
            a float value specifying text thickness
        BIO_STYLE (tuple): Specifies the style for the text containing
            the identity age and gender. It is a tuple containing cv2
            enumerator for font face, a float value specifying the text
            scale, a tuple specifying RGB color and a float value
            specifying text thickness
    """
    BOX_STYLE = ((0, 0, 255), 2)
    MARKS_STYLE = ([(0, 255, 0), (0, 0, 255), (255, 0, 0)], 2)
    NAME_STYLE = (cv2.FONT_HERSHEY_COMPLEX, 0.45, (75, 255, 0), 1)
    BIO_STYLE = (cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 40), 1)

    def __init__(self, drawable):
        """Initializes the view handler

        Simply assigns the list of drawable items as this class'
        attribute.

        Args:
            drawable (list(str)): The list of desired items to draw on
                the image with the recognized face(-s). It can contain
                the following values: ["box", "marks", "name", "bio"]
        """
        self.drawable = drawable
    
    def draw_on_frame(self, frame, identities):
        """Draws boxes, marks, names, bio information on a raw image

        Takes the frame (image) and identities dictionary with values
        for "boxes", "marks", "names", "scores", "genders" and "ages"
        for each identity and draws what is specified in `self.drawable`
        on the taken frame.

        Args:
            frame (numpy.ndarray): An image expressed as a numpy array
            identities (dict): The dictionary with keys indicating
                features, such as "names", "genders", and values
                indicating identities, i.e, lists where each entry is a
                feature that the key specifies

        Returns:
            numpy.ndarray: A modified frame with drawn items on it
        """
        if identities is None or len(identities) == 0:
            # No changes
            return frame
        
        # Work on a copy
        img = frame.copy()
        
        for i in range(len(identities["boxes"])):
            # Get current i^th identity
            bbox = identities["boxes"][i]
            marks = identities["marks"][i]
            name = identities["names"][i]
            score = identities["scores"][i]
            gender = identities["genders"][i]
            age = identities["ages"][i]
            
            # Text coordinates
            x1 = x2 = bbox[0]
            y1 = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
            y2 = bbox[3] + 20 if bbox[3] + 20 > 20 else bbox[3] - 20

            # Draw whatever's desired
            self.draw_box(img, bbox)
            self.draw_marks(img, marks)
            self.draw_name(img, name, score, (x1, y1))
            self.draw_bio(img, gender, age, (x2, y2))
        
        return img
    
    def draw_box(self, img, box):
        """Draws a boundary box around the face

        Takes and image on which to draw the box and the coordinates and
        2 opposite corner vertices, and draws a bounding box.

        Args:
            img (numpy.ndarray): The image represented as a numpy array
            box (list(int)): The list containing 4 values: [x1, y1, x2,
                y2] where (x1, y1) - coordinate for the top left corner
                of the bounding box, (x2, y2) - coordinate for the
                bottom right corner
        """
        if "box" not in self.drawable:
            return
        
        # Just draw a rectangle (boundary box) around the detected face
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), *self.BOX_STYLE)
    
    def draw_marks(self, img, marks):
        """Draws 5 landmarks on the face

        Takes an image and a list of landmark coordinates. It loops
        through all 5 coordinates and draws them on the face: two on
        the left, two on the right and one in the middle. 

        Args:
            img (numpy.ndarray): The image represented as a numpy array
            marks (list(list(int))): A list of lists of coordinate
                values that specify landmark positions. Each sublist in
                the list has 2 coordinate values - x and y.
        """
        if "marks" not in self.drawable:
            return
        
        for i, mark in enumerate(marks):
            if i == 0 or i == 3:
                # Color for up/down left marks
                color = self.MARKS_STYLE[0][0]
            elif i == 1 or i == 4:
                # Color for up/down right marks
                color = self.MARKS_STYLE[0][1]
            else:
                # Color for middle (nose) mark
                color = self.MARKS_STYLE[0][2]
            
            # Draw small filled circle representing the current landmark
            cv2.circle(img, (mark[0], mark[1]), 1, color, self.MARKS_STYLE[1])
    
    def draw_name(self, img, name, prob, coords):
        """Writes the name and the confidence score near the face

        Takes an image, a name, probability (confidence score) and
        coordinates of where to draw the full text and places the
        generated text there.

        Args:
            img (numpy.ndarray): The image represented as a numpy array
            name (str): The name of the recognized person (face)
            prob (float): The probability score between 0 and 1
            coords (tuple(int)): The tuple representing x and y
                coordinate values specifying where to place the text
        """
        if "name" not in self.drawable:
            return

        # Join text and draw at coordinates
        text = f"{name} ({prob*100:.2f}%)"
        cv2.putText(img, text, coords, *self.NAME_STYLE)
    
    def draw_bio(self, img, gender, age, coords):
        """Writes the age and the gender near the face

        Takes an image, a gender type, age and coordinates of where to
        draw the full text and places the generated text there.

        Args:
            img (numpy.ndarray): The image represented as a numpy array
            gender (bool): The gender type (True - male, False - female)
            age (int): The age of the person (face)
            coords (tuple(int)): The tuple representing x and y
                coordinate values specifying where to place the text
        """
        if "bio" not in self.drawable:
            return
        
        # Join text and draw at specified coordinates
        text = f"{'Male' if gender else 'Female'}, {age}"
        cv2.putText(img, text, coords, *self.BIO_STYLE)