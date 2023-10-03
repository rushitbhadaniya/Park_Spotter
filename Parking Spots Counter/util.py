import pickle
import cv2
from skimage.transform import resize
import numpy as np

NOT_PARKED= True
PARKED= False

MODEL= pickle.load(open("model.p",'rb'))

def parked_or_not(spot_rgb):
    """
        Determines whether a parking spot is parked or not based on the provided RGB image.

        Parameters:
        spot_rgb: The RGB image of the parking spot to be evaluated.

        Returns:
        str: Returns "PARKED" if the parking spot is occupied, "NOT_PARKED" if it's empty.

        This function takes an RGB image of a parking spot, resizes and flattens the image, and
        uses a machine learning model to predict whether the spot is parked or not. If the
        prediction is 0, it indicates the spot is empty and returns "NOT_PARKED." If the
        prediction is non-zero, it indicates the spot is occupied and returns "PARKED."
        """
    flat_data=[]

    img_resized= resize(spot_rgb,(15,15,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)

    y_output= MODEL.predict(flat_data)

    if y_output ==0:
        return NOT_PARKED
    else:
        return  PARKED


def get_parking_spots_bbox(connected_components):
    """
        Extracts bounding boxes for parking spots from connected components.

        Parameters:
        connected_components (tuple): A tuple containing information about connected components,
            typically obtained through a connected components labeling algorithm.
            The tuple structure is (totalLabels, label_idx, values, centriod).

        Returns:
        list: A list of bounding boxes (rectangles) for parking spots.
            Each bounding box is represented as a list [x1, y1, width, height].

        This function takes connected components information and extracts bounding boxes
        for parking spots. It processes the connected components, scales the coordinates,
        and returns a list of bounding boxes for parking spots found in the image.
        """
    (totalLabels, label_idx, values, centriod) = connected_components
    slot=[]

    coef=1

    for i in range(1, totalLabels):

        #now Extract the coordinate points
        x1 = int(values[i,cv2.CC_STAT_LEFT]* coef)
        y1 = int(values[i,cv2.CC_STAT_TOP]* coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slot.append([x1,y1,w,h])

    return slot

def calc_diff(im1,im2):
    """
        Calculate the absolute difference in the mean values between two images.

        Parameters:
        im1: The first input image.
        im2: The second input image.

        Returns:
        float: The absolute difference in the mean values between the two images.

        This function calculates the absolute difference in the mean values of two images.
        It provides a measure of dissimilarity between the images based on their average pixel intensity.
        """
    return np.abs(np.mean(im1)-np.mean(im2))
