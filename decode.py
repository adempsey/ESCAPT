import numpy as np
import cv2
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
import os, sys

NUM_CHARACTERS = 5
IMAGE_SIZE = 30

def loadTrainingExamples():
    files = Popen("ls train/", shell=True, stdout=PIPE).stdout.read()
    files = files.strip().split('\n')

    examples = []
    labels = []

    for f in files:
        labels.append(f[0])
        example = cv2.imread(os.path.join("train",f))
        example = cv2.resize(example, (IMAGE_SIZE,IMAGE_SIZE))
        example, exampleContours = processImage(example)
        examples.append(example)

    return examples, labels

def loadTestExample(filename):
    image = cv2.imread(filename)

    # We apply a modified kernel to the image - coefficients were obtained
    # after trial and error to see what produced the best result
    kernel = np.ones((3,2), np.uint8)
    image, contours = processImage(image, kernel)

    contours = filterTestSegments(contours)
    return image, contours

def removeNoise(image):
    return cv2.fastNlMeansDenoisingColored(image,None,13,13,5,41)

# Obtain black and white threshold of image for segmentation
def threshold(image, kernel=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 1, 11, 2)
    f,image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if kernel != None:
        thresh = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours

# We know that each CAPTCHA contains 5 characters, so we choose the five
# largest contours and sort them
def filterTestSegments(contours):
    rectangles = map(lambda c: (c,cv2.boundingRect(c)), contours)
    rectangles = map(lambda (c,(x,y,w,h)): (c, (x,y,w,h), w*h), rectangles)
    rectangles = sorted(rectangles, key=lambda tup: tup[2], reverse=True)[:NUM_CHARACTERS]
    rectangles = sorted(rectangles, key=lambda tup: tup[1][0], reverse=False)
    return [c[0] for c in rectangles]

# Return a new subimage from the parent image defined by the contour
def subRectangle(image, contour):
    x,y,w,h = cv2.boundingRect(contour)
    image = image[y:y+h,x:x+w]
    image = abs(255-image)
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
    return image

def similarity(image1, image2):
    diff = image1 - image2
    return (IMAGE_SIZE*IMAGE_SIZE) - cv2.countNonZero(diff)

# Determine the label with the highest similarity in the list
def maxSimilarity(vals):
    vals = sorted(vals, key=lambda tup:tup[0], reverse=False)
    return vals[0][1]

# Remove noise and obtain contours
def processImage(image, kernel=None):
    image = removeNoise(image)
    return threshold(image, kernel)

def printResults(actuals, answers):
    print "GUESS:\t",
    for a in answers:
        print a,
    print ""
    print "\t",
    for (actual, answer) in zip(actuals, answers):
        print u'\u2713' if actual == answer else u'\u2717',
    print ""
    print "ACTUAL:\t",
    for a in actuals:
        print a,
    print ""
    print "-----------------"

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: python decode.py <testfile 1> ... <testfile n>"
        print "Filenames should represent actual CAPTCHA values for comparison"
        exit()

    trainingExamples, labels = loadTrainingExamples()

    for testFile in sys.argv[1:]:
        img, contours = loadTestExample(testFile)

        answers = []

        for c in contours:
            testImage = subRectangle(img, c)

            results = []
            for trainImage, label in zip(trainingExamples,labels):
                results.append((similarity(trainImage, testImage), label))

            answers.append(maxSimilarity(results))

        actuals = list(testFile.split('/')[-1].split('.')[0])

        printResults(actuals, answers)
