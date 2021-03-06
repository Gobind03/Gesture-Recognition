import cv2
import numpy as np
import matplotlib.pyplot as plt

def referTemplate(iFrame, threshold):
    
    img_gray = cv2.cvtColor(iFrame, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('train.jpg',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(iFrame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

    return iFrame

def bruteForceMatcher(iFrame):
    
    img1 = cv2.imread('train.jpg',0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(iFrame,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
    except MatchErrorException as e:
        print "No or Improbable Match Found"

    img3 = cv2.drawMatches(img1,kp1,iFrame,kp2,matches[:10],None, flags=2)
    
    #plt.imshow(img3)
    #plt.show()
    
    return img3


