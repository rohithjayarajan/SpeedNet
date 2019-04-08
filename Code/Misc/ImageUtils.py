"""
@file    ImageUtils.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import random
import tensorflow as tf

debug = False


class ImageUtils:

    def PreProcess(self, InputImage, ResizeHeight, ResizeWidth):
        BlurImage = cv2.GaussianBlur(InputImage, (5, 5), 0)
        GrayImage = cv2.cvtColor(BlurImage, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        HistEqImage = clahe.apply(GrayImage)
        CroppedImg = cv2.resize(HistEqImage, dsize=(
            ResizeHeight, ResizeWidth), interpolation=cv2.INTER_CUBIC)
        return CroppedImg

    def ImageStandardizationColor(self, InputImage):
        # mu = np.mean(InputImage)
        # StandardizedImage = (InputImage - mu)/float(255)
        StandardizedImage = cv2.normalize(
            InputImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return StandardizedImage

    def ImageStandardizationBW(self, InputImage):
        mu = np.mean(InputImage)
        StandardizedImage = (InputImage - mu)/float(255)
        return StandardizedImage

    def CreateImgStackData(self, InputImagePrev, InputImageNext):
        InputImagePrev = np.float32(InputImagePrev)
        InputImageNext = np.float32(InputImageNext)
        InputImagePrev = self.ImageStandardizationBW(InputImagePrev)
        InputImageNext = self.ImageStandardizationBW(InputImageNext)
        return tf.concat([InputImagePrev, InputImageNext], 0)

    def CreateFlowData(self, InputImagePrev, InputImageNext, hsv):
        hsv[..., 1] = 255
        flow = cv2.calcOpticalFlowFarneback(
            InputImagePrev, InputImageNext, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # print("hsv.shape: {}".format(hsv.shape))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
