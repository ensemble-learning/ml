#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yufeng Huang
"""

import tensorflow as tf
import tf_func as tff


def scaleFeat(featParams, feat):
    return featParams['featA'] * feat + featParams['featB']


def trainEL_getError(featSets, engySets, featParams, nnParams, logFile="log"):
    AdFeatTrain, AdFeatValid, AdFeatTest, _, __, ___ = featSets
    AdEngyTrain, AdEngyValid, AdEngyTest, _, __, ___ = engySets

    tf_feat = tf.placeholder(tf.float32, (None, featParams['nFeat']))
    tf_engy = tf.placeholder(tf.float32, (None))
    tf_LR = tf.placeholder(tf.float32)

    L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)

    engyLoss = tf.reduce_mean((L3 - tf_engy) ** 2)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./" + logFile + "/model.ckpt")

        AdTrainDict = {tf_feat: scaleFeat(featParams, AdFeatTrain), tf_engy: AdEngyTrain,
                       tf_LR: nnParams['learningRate']}
        AdTestDict = {tf_feat: scaleFeat(featParams, AdFeatTest), tf_engy: AdEngyTest,
                      tf_LR: nnParams['learningRate']}
        AdValidDict = {tf_feat: scaleFeat(featParams, AdFeatValid), tf_engy: AdEngyValid,
                       tf_LR: nnParams['learningRate']}

        eLoss = sess.run(engyLoss, feed_dict=AdTrainDict)
        veLoss = sess.run(engyLoss, feed_dict=AdTestDict)
        teLoss = sess.run(engyLoss, feed_dict=AdValidDict)

        print("Training set RMSE: {:10.4f} eV".format(eLoss**0.5))
        print("Validation set RMSE: {:10.4f} eV".format(veLoss**0.5))
        print("Testing set RMSE: {:10.4} eV".format(teLoss**0.5))


def getE(feat, featParams, nnParams, logFile="log"):
    tf_feat = tf.placeholder(tf.float32, (None, featParams['nFeat']))
    L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./" + logFile + "/model.ckpt")

        feedDict = {tf_feat: scaleFeat(featParams, feat)}

        engy = sess.run(L3, feed_dict=feedDict)

    return engy
