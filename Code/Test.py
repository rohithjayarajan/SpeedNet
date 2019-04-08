import tensorflow as tf
import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import SpeedNetwork
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.OpticalFlowDense import GunnerFarnebackFlow
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Misc.ImageUtils import *
from Misc.HelperFunctions import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


class Test:

    def __init__(self, NumTrainSamples, LatestFile, DirNamesTrain, TrainLabels, DirNamesVal, ValLabels, ImageSize, SaveCheckPoint, ModelPath, BasePath, LogsPath, ModelType):
        self.BasePath = BasePath
        self.MiniBatchSize = 2399
        self.NumTrainSamples = NumTrainSamples
        self.TrainLabels = TrainLabels
        self.DirNamesVal = DirNamesVal
        self.ValLabels = ValLabels
        self.LatestFile = LatestFile
        self.ImgPH = tf.placeholder('float', shape=(2398, 66, 200, 3))
        self.LabelPH = tf.placeholder('float', shape=(2398, 1))
        self.DirNamesTrain = DirNamesTrain
        self.ImageSize = [66, 200, 3]
        self.SaveCheckPoint = SaveCheckPoint
        self.ModelPath = ModelPath
        self.LogsPath = LogsPath
        self.ModelType = ModelType
        self.Model = SpeedNetwork()
        self.ImageUtils = ImageUtils()
        self.HelperFunctions = HelperFunctions()

    def GenerateVal(self):
        I1Batch = []
        LabelBatch = []

        ImageNum = 0
        while ImageNum < self.MiniBatchSize-1:

            RandImageName1 = self.BasePath + os.sep + \
                self.DirNamesVal[ImageNum] + '.jpg'
            RandImageName2 = self.BasePath + os.sep + \
                self.DirNamesVal[ImageNum+1] + '.jpg'
            ImageNum += 1

            ##########################################################
            # Add any standardization or data augmentation here!
            ##########################################################

            Im1 = (cv2.imread(RandImageName1))
            Im2 = (cv2.imread(RandImageName2))

            # Imt = cv2.resize(Im1, dsize=(224, 224))
            Imt = cv2.resize(Im1, dsize=(200, 66))
            hsv = np.zeros_like(Imt)

            # Im1 = self.ImageUtils.PreProcess(Im1, 224, 224)
            # Im2 = self.ImageUtils.PreProcess(Im2, 224, 224)
            Im1 = self.ImageUtils.PreProcess(Im1, 200, 66)
            Im2 = self.ImageUtils.PreProcess(Im2, 200, 66)

            # self.HelperFunctions.ShowImage(Im1, "Im1")
            # self.HelperFunctions.ShowImage(Im2, "Im2")

            flow = self.ImageUtils.CreateFlowData(Im1, Im2, hsv)
            # self.HelperFunctions.ShowImage(flow, "flowb4")

            flow = np.float32(flow)
            flow = self.ImageUtils.ImageStandardizationColor(flow)
            # self.HelperFunctions.ShowImage(flow, "flow")

            # flow = self.ImageUtils.CreateImgStackData(Im1, Im2)
            Label = self.ValLabels[ImageNum+1]
            # Append All Images and Mask
            I1Batch.append(flow)
            # I1Batch.append(stackIm)
            LabelBatch.append(Label)

        LabelBatch = np.reshape(LabelBatch, (-1, 1))
        return I1Batch, LabelBatch

    def TestOperation(self):

        # SpeedPt = self.Model.SpeedNetVGG(self.ImgPH, False)
        SpeedPt = self.Model.SpeedNetNVIDIAe2e(self.ImgPH, False)

        with tf.name_scope('Loss'):
            # loss = tf.reduce_mean(tf.math.square(
            #     SpeedPt - self.LabelPH))
            loss = tf.losses.mean_squared_error(SpeedPt, self.LabelPH)

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            Saver.restore(sess, self.ModelPath)

            I1Batch, LabelBatch = self.GenerateVal()
            FeedDict = {self.ImgPH: I1Batch}
            PredSpeedPt = sess.run(SpeedPt, feed_dict=FeedDict)
            # LossThisBatch /= self.MiniBatchSize
            print("loss: {}".format(((PredSpeedPt - LabelBatch)**2).mean()))


def main():
        # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/rohith/CMSC733/git/SpeedNet/Data',
                        help='Base path of training video, Default:/home/rohith/CMSC733/git/SpeedNet/Data')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/rohith/CMSC733/git/SpeedNet/Checkpoints/17model.ckpt',
                        help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ModelType', default='Unsup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=20,
                        help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1,
                        help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=2399,
                        help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0,
                        help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/',
                        help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    ModelPath = Args.ModelPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, DirNamesVal, ValLabels = SetupAll(
        BasePath, ModelPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(ModelPath)
    else:
        LatestFile = None

    Trainer = Test(NumTrainSamples, LatestFile, DirNamesTrain, TrainLabels, DirNamesVal,
                   ValLabels, ImageSize, SaveCheckPoint, ModelPath, BasePath, LogsPath, ModelType)

    Trainer.TestOperation()


if __name__ == '__main__':
    main()
