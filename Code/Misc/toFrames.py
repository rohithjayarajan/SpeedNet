import cv2

videoPath = '/home/rohith/CMSC733/git/SpeedNet/Data/train.mp4'
framesPath = '/home/rohith/CMSC733/git/SpeedNet/Data/TrainImages/'
vid = cv2.VideoCapture(videoPath)
frameNumber = 1

while(vid.isOpened()):
    ret, frame = vid.read()

    if ret == True:
        cv2.imwrite('%s' % (framesPath) + "/%#08d.jpg" %
                    (frameNumber), frame)
        frameNumber += 1
        print(frameNumber)

    else:
        break

vid.release()
cv2.destroyAllWindows()
