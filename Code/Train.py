#!/usr/bin/env python

"""
@file    Train.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

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


class Train:

    def __init__(self, NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile, ImgPH, LabelPH, DirNamesTrain, TrainLabels, DirNamesVal, ValLabels, ImageSize, SaveCheckPoint, CheckPointPath, BasePath, LogsPath, ModelType):
        self.NumEpochs = NumEpochs
        self.DivTrain = DivTrain
        self.BasePath = BasePath
        self.MiniBatchSize = MiniBatchSize
        self.NumTrainSamples = NumTrainSamples
        self.TrainLabels = TrainLabels
        self.DirNamesVal = DirNamesVal
        self.ValLabels = ValLabels
        self.LatestFile = LatestFile
        self.ImgPH = ImgPH
        self.LabelPH = LabelPH
        self.DirNamesTrain = DirNamesTrain
        self.ImageSize = ImageSize
        self.SaveCheckPoint = SaveCheckPoint
        self.CheckPointPath = CheckPointPath
        self.LogsPath = LogsPath
        self.ModelType = ModelType
        self.Model = SpeedNetwork()
        self.ImageUtils = ImageUtils()
        self.HelperFunctions = HelperFunctions()

    def GenerateVal(self):
        I1Batch = []
        LabelBatch = []

        ImageNum = 0
        while ImageNum < self.MiniBatchSize:
            RandIdx = random.randint(0, len(self.DirNamesVal)-2)
            RandImageName1 = self.BasePath + os.sep + \
                self.DirNamesVal[RandIdx] + '.jpg'
            RandImageName2 = self.BasePath + os.sep + \
                self.DirNamesVal[RandIdx+1] + '.jpg'
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
            Label = self.ValLabels[RandIdx+1]
            # Append All Images and Mask
            I1Batch.append(flow)
            # I1Batch.append(stackIm)
            LabelBatch.append(Label)

        LabelBatch = np.reshape(LabelBatch, (-1, 1))
        return I1Batch, LabelBatch

    def GenerateBatch(self):
        """
        Inputs: 
        BasePath - Path to COCO folder without "/" at the end
        DirNamesTrain - Variable with Subfolder paths to train files
        NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
        TrainLabels - Labels corresponding to Train
        NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
        ImageSize - Size of the Image
        MiniBatchSize is the size of the MiniBatch
        Outputs:
        I1Batch - Batch of images
        LabelBatch - Batch of one-hot encoded labels 
        """
        I1Batch = []
        LabelBatch = []

        ImageNum = 0
        while ImageNum < self.MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(self.DirNamesTrain)-2)

            RandImageName1 = self.BasePath + os.sep + \
                self.DirNamesTrain[RandIdx] + '.jpg'
            RandImageName2 = self.BasePath + os.sep + \
                self.DirNamesTrain[RandIdx+1] + '.jpg'
            ImageNum += 1

            ##########################################################
            # Add any standardization or data augmentation here!
            ##########################################################

            Im1 = (cv2.imread(RandImageName1))
            Im2 = (cv2.imread(RandImageName2))
            RandAug = random.uniform(0, 1)

            if RandAug > 0 and RandAug < 0.5:
                Im1 = np.fliplr(Im1)
                Im2 = np.fliplr(Im2)

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
            Label = self.TrainLabels[RandIdx+1]
            # Append All Images and Mask
            I1Batch.append(flow)
            # I1Batch.append(stackIm)
            LabelBatch.append(Label)

        LabelBatch = np.reshape(LabelBatch, (-1, 1))
        return I1Batch, LabelBatch

    def PrettyPrint(self):
        """
        Prints all stats with all arguments
        """
        print('Number of Epochs Training will run for ' + str(self.NumEpochs))
        print('Factor of reduction in training data is ' + str(self.DivTrain))
        print('Mini Batch Size ' + str(self.MiniBatchSize))
        print('Number of Training Images ' + str(self.NumTrainSamples))
        if self.LatestFile is not None:
            print('Loading latest checkpoint with the name ' + self.LatestFile)

    def TrainOperation(self):
        """
        Inputs: 
        ImgPH is the Input Image placeholder
        LabelPH is the one-hot encoded label placeholder
        DirNamesTrain - Variable with Subfolder paths to train files
        TrainLabels - Labels corresponding to Train/Test
        NumTrainSamples - length(Train)
        ImageSize - Size of the image
        NumEpochs - Number of passes through the Train data
        MiniBatchSize is the size of the MiniBatch
        SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
        CheckPointPath - Path to save checkpoints/model
        DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
        LatestFile - Latest checkpointfile to continue training
        BasePath - Path to COCO folder without "/" at the end
        LogsPath - Path to save Tensorboard Logs
            ModelType - Supervised or Unsupervised Model
        Outputs:
        Saves Trained network in CheckPointPath and Logs to LogsPath
        """

        # Predict output with forward pass
        # SpeedPt = self.Model.SpeedNetVGG(self.ImgPH, True)
        SpeedPt = self.Model.SpeedNetNVIDIAe2e(self.ImgPH, True)

        with tf.name_scope('Loss'):
            # loss = tf.reduce_mean(tf.math.square(
            #     SpeedPt - self.LabelPH))
            loss = tf.losses.mean_squared_error(SpeedPt, self.LabelPH)

        with tf.name_scope('Adam'):
            Optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        LossPerEpoch = tf.placeholder(tf.float32, shape=None)

        # Tensorboard
        # Create a summary to monitor loss tensor
        IterLoss = tf.summary.scalar('LossEveryIter', loss)
        ValLoss = tf.summary.scalar('ValLossEveryIter', loss)
        EpochLoss = tf.summary.scalar('LossEveryEpoch', LossPerEpoch)
        IterSpeed = tf.summary.histogram('SpeedPerIter', SpeedPt)
        # WarpedImage = tf.summary.image('WarpedImageSummary', WarpedIm)
        # UnWarpedImage = tf.summary.image('UnWarpedImageSummary', UnWarpedIm)
        # tf.summary.image('Anything you want', AnyImg)
        # Merge all summaries into a single operation
        MergedSummaryPerIter = tf.summary.merge(
            [IterLoss, IterSpeed])
        ValSummaryPerIter = tf.summary.merge(
            [ValLoss])
        MergedSummaryPerEpoch = tf.summary.merge(
            [EpochLoss])

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            if self.LatestFile is not None:
                Saver.restore(sess, self.CheckPointPath +
                              self.LatestFile + '.ckpt')
                # Extract only numbers from the name
                StartEpoch = int(
                    ''.join(c for c in self.LatestFile.split('a')[0] if c.isdigit()))
                print('Loaded latest checkpoint with the name ' +
                      self.LatestFile + '....')
            else:
                sess.run(tf.global_variables_initializer())
                StartEpoch = 0
                print('New model initialized....')

            # Tensorboard
            Writer = tf.summary.FileWriter(
                self.LogsPath, graph=tf.get_default_graph())

            for Epochs in tqdm(range(StartEpoch, self.NumEpochs)):
                NumIterationsPerEpoch = int(
                    self.NumTrainSamples/self.MiniBatchSize/self.DivTrain)
                LossPerEpochVar = 0
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    I1Batch, LabelBatch = self.GenerateBatch()
                    FeedDict = {self.ImgPH: I1Batch, self.LabelPH: LabelBatch}
                    _, LossThisBatch, IterSummary = sess.run(
                        [Optimizer, loss, MergedSummaryPerIter], feed_dict=FeedDict)
                    LossPerEpochVar += LossThisBatch

                    # Tensorboard
                    Writer.add_summary(
                        IterSummary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                LossPerEpochVar /= NumIterationsPerEpoch
                FeedDict2 = {LossPerEpoch: LossPerEpochVar}
                I1Val, LabelVal = self.GenerateVal()
                FeedDictVal = {self.ImgPH: I1Val, self.LabelPH: LabelVal}
                _, IterSummary = sess.run(
                    [loss, ValSummaryPerIter], feed_dict=FeedDictVal)
                EpochSummary = sess.run(
                    MergedSummaryPerEpoch, feed_dict=FeedDict2)
                Writer.add_summary(EpochSummary, Epochs)
                Writer.add_summary(IterSummary, Epochs)
                Writer.flush()

                # Save model every epoch
                if Epochs != 0 and (Epochs > 25):
                    SaveName = self.CheckPointPath + str(Epochs) + 'model.ckpt'
                    Saver.save(sess, save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

            SaveName = self.CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/rohith/CMSC733/git/SpeedNet/Data',
                        help='Base path of training video, Default:/home/rohith/CMSC733/git/SpeedNet/Data')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=75,
                        help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1,
                        help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=512,
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
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, DirNamesVal, ValLabels = SetupAll(
        BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(
        MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.placeholder(tf.float32, shape=(
        MiniBatchSize, 1))

    Trainer = Train(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile, ImgPH, LabelPH,
                    DirNamesTrain, TrainLabels, DirNamesVal, ValLabels, ImageSize, SaveCheckPoint, CheckPointPath, BasePath, LogsPath, ModelType)

    # Pretty print stats
    Trainer.PrettyPrint()

    Trainer.TrainOperation()


if __name__ == '__main__':
    main()
