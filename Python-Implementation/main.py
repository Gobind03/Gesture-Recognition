import cv2
import numpy as np
import ObjectRecognition as orgnt

frame = cv2.imread('base.jpg')
pframe = orgnt.referTemplate(frame)
cv2.imwrite('res.png',pframe)

x = raw_input()
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     pframe = orgnt.referTemplate(frame)
#     # Display the resulting frame
#     cv2.imshow('frame',pframe)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()