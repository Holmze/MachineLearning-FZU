#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: Shiping Wang
@ Email: shipingwangphd@163.com
"""

import AdaboostNavieBayes as boostNaiveBayes
from sklearn import preprocessing
import numpy as np


def Text2Vector( ):
    """
    return: feature matrix: nxd
            labels:  n x 1
    """

    ### Step 1: Read data 
    filename = './emails/training/SMSCollection.txt'
    smsWords, classLabels = boostNaiveBayes.loadSMSData(filename)
    classLabels = np.array(classLabels)


    ### STEP 2: Transform the original data into feature matrix
    vocabularyList = boostNaiveBayes.createVocabularyList(smsWords)
    print("生成语料库！")
    trainMarkedWords = boostNaiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print("数据标记完成！")
    # 转成array向量
    trainMarkedWords = np.array(trainMarkedWords)  ### Traning feature matrix N x d
    #print("The all feature matrix size is: ", trainMarkedWords.shape)
    
    return trainMarkedWords, classLabels


if __name__ == '__main__':
    
    trainMarkedWords, classLabels = Text2Vector()
    
