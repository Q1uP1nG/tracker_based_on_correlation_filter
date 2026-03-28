import cv2
import sys
from time import time

import basetracker

initTracking = True
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
inteval = 1
duration = 0.01

if __name__ == '__main__':

    # Read in camera input or video
    if (len(sys.argv) == 1):
        cap = cv2.VideoCapture(0)
    elif (len(sys.argv) == 2):
        if (sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a non-negative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
            inteval = 30
    else:
        assert (0), "too many arguments"

    # Todo: Implement the KCF Tracker by yourself
    # tracker = kcftracker.KCFTracker(False, True, False)
    tracker = basetracker.BaseTracker(False, True, False)  # hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox,
    # that is due to the use of Numba.

    cv2.namedWindow('tracking')

    # Iteration over whole video frames
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if (initTracking):
            # Capture the bounding box selected by the user
            r = cv2.selectROI('tracking', frame, False, False)
            ix, iy, w, h = r
            tracker.init([ix, iy, w, h], frame)
            initTracking = False  # init tracker only once
        else:
            # Update the bounding box according to new frame
            t0 = time()
            boundingbox = tracker.update(frame)
            t1 = time()

            # Draw the new bounding box
            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(frame,  # the image on which rectangle is to be drawn
                          (boundingbox[0], boundingbox[1]),  # start point
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),  # end point
                          (0, 255, 255),  # color
                          1)  # thickness

            # Show the time duration
            duration = 0.8 * duration + 0.2 * (t1 - t0)      # moving average
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        # Show the result in the window
        cv2.imshow('tracking', frame)
        # Press q to quit
        c = cv2.waitKey(inteval) & 0xFF   
        if c == 27 or c == ord('q'):  # 27是"esc"的ascii码
            break

    # Finish tracking
    cap.release()
    cv2.destroyAllWindows()
