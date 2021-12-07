import cv2
import random
import time
import os
import sys
from PIL import Image
import numpy as np

# load the pre-trained model
classifier = cv2.CascadeClassifier("model_cascade.xml")

# Paths
BASE_PATH = "../dataset"
IMAGES_PATH = os.path.join(BASE_PATH, "faces_aligned_small_mirrored_co_aligned_cropped_cleaned", "F")
SINGULAR_TEST_PATH = "../images/data/people.jpg"
COLLAGE_SIZE = 30
COLLAGE_SHAPE = 6


dataset = []

"""
We create collages to challenge the cascader a bit.
"""

def show_images(dataset):
    for image in dataset:
        print(type(image))
        cv2.imshow("Output", image)
        key = cv2.waitKey(0)
        if key == 27:
            break

def prepare_dataset():
    print("*** Preparing collage dataset")
    num_collages = 0
    if(os.path.exists(IMAGES_PATH)):
        print("The path used to collect images", IMAGES_PATH)
        for i in range(0, len(os.listdir(IMAGES_PATH)), 6):
            filename1 = os.listdir(IMAGES_PATH)[i]
            filename2 = os.listdir(IMAGES_PATH)[i+1]
            filename3 = os.listdir(IMAGES_PATH)[i+2]
            filename4 = os.listdir(IMAGES_PATH)[i+3]
            filename5 = os.listdir(IMAGES_PATH)[i+4]
            filename6 = os.listdir(IMAGES_PATH)[i+5]
            
            if filename1.endswith(".png") and filename2.endswith(".png") and filename3.endswith(".png") and filename4.endswith(".png") and filename5.endswith(".png") and filename6.endswith(".png"): 
                image1 = cv2.imread(os.path.join(IMAGES_PATH, filename1), cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(os.path.join(IMAGES_PATH, filename2), cv2.IMREAD_GRAYSCALE)
                image3 = cv2.imread(os.path.join(IMAGES_PATH, filename3), cv2.IMREAD_GRAYSCALE)
                image4 = cv2.imread(os.path.join(IMAGES_PATH, filename4), cv2.IMREAD_GRAYSCALE)
                image5 = cv2.imread(os.path.join(IMAGES_PATH, filename5), cv2.IMREAD_GRAYSCALE)
                image6 = cv2.imread(os.path.join(IMAGES_PATH, filename6), cv2.IMREAD_GRAYSCALE)  
                col_1 = np.vstack([image1, image2])
                col_2 = np.vstack([image3, image4])
                col_3 = np.vstack([image5, image6])
                collage = np.hstack([col_1, col_2, col_3])
                dataset.append(collage)
                num_collages = num_collages + 6
                print("*** ", i, "th collage made!" )
                if(num_collages >=COLLAGE_SIZE):
                    print("*** Done !")
                    break
            else:
                continue   
    else:
        print("The path does not exist !")
        
def test_model(test_dataset = dataset):
    print("*** Operating collage dataset")
    for image in test_dataset:
        bboxes = classifier.detectMultiScale(image)
        #Convert the image to color (otherwise the boxes will be black)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print bounding box for each detected face
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            cv2.rectangle(image_bgr, (x, y), (x2, y2), (255,0,0), 1)
        cv2.imshow("BBX", image_bgr)
        key = cv2.waitKey(0)
        if key == 27:
            break
        
if __name__ == '__main__':       
    #prepare_dataset()
    #test_model()
    
    #Singular test with background
    img = cv2.imread(SINGULAR_TEST_PATH, cv2.IMREAD_GRAYSCALE)
    test_model([img])























































    