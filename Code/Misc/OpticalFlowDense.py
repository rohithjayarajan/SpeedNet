#!/usr/bin/env python

"""
@file    OpticalFlowDense.py
@author  rohithjayarajan
@date 04/04/2019

Licensed under the
GNU General Public License v3.0
"""

import cv2
import numpy as np
import argparse


class GunnerFarnebackFlow:
    def __init__(self, TrainVideo):
        self.TrainVideo = TrainVideo
        self.BBox = [0, 35, 640, 310]

    def CropImage(self, Image):
        CroppedImage = Image[int(self.BBox[1]):int(self.BBox[1]+self.BBox[3]),
                             int(self.BBox[0]):int(self.BBox[0]+self.BBox[2])]
        return CroppedImage

    def ComputeFlow(self):
        Cap = cv2.VideoCapture(self.TrainVideo)
        Ret, Frame1 = Cap.read()
        # print(Frame1.shape)
        Frame1 = self.CropImage(Frame1)
        hsv = np.zeros_like(Frame1)
        hsv[..., 1] = 255
        PrevImageGray = cv2.cvtColor(Frame1, cv2.COLOR_BGR2GRAY)
        f = 1

        while(1):
            Ret, Frame2 = Cap.read()
            Frame2 = self.CropImage(Frame2)
            f += 1

            if(f % 15000 == 0):
                print(f)
                cv2.imshow('Frame2', Frame1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            NextImageGray = cv2.cvtColor(Frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                PrevImageGray, NextImageGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # print(flow.shape)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            print("hsv.shape: {}".format(hsv.shape))
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('Frame2', bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # cv2.imwrite('opticalfb.png', Frame2)
            # cv2.imwrite('opticalhsv.png', bgr)
            PrevImageGray = NextImageGray
        Cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Inputs:
    Path to train.mp4

    Outputs:
    Computes dense optical flow between pair of images
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--TrainVideo', default='/home/rohith/CMSC733/git/SpeedNet/Data/train.mp4',
                        help='Base path of training video, Default:/home/rohith/CMSC733/git/SpeedNet/Data/train.mp4')
    Args = Parser.parse_args()
    TrainVideo = Args.TrainVideo

    DenseFlow1 = GunnerFarnebackFlow(TrainVideo)
    DenseFlow1.ComputeFlow()


if __name__ == '__main__':
    main()
