from __future__ import print_function
import os
import sys
import numpy as np
import pickle as cp
import cifar_utils as ut

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("Usage: CifarConverter.py <path to CIFAR-10 dataset directory>\nCIFAR-10 dataset (Python version) can be downloaded from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        sys.exit(1)
    rootDir = sys.argv[1]
    trainDir = 'train'
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    testDir = 'test'
    if not os.path.exists(testDir):
      os.makedirs(testDir)
    data = {}
    dataMean = np.zeros((3, ut.ImgSize, ut.ImgSize)) # mean is in CHW format.
    with open('train_map.txt', 'w') as mapFile:
        with open('train_regrLabels.txt', 'w') as regrFile:
            for ifile in range(1, 6):
                with open(os.path.join(rootDir, 'data_batch_' + str(ifile)), 'rb') as f:
                    if sys.version_info[0] < 3: 
                        data = cp.load(f)
                    else: 
                        data = cp.load(f, encoding='latin1')
                    for i in range(10000):
                        fname = os.path.join(os.path.abspath(trainDir), ('%05d.png' % (i + (ifile - 1) * 10000)))
                        ut.saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    dataMean = dataMean / (50 * 1000)
    ut.saveMean('CIFAR10_mean.xml', dataMean)
    with open('test_map.txt', 'w') as mapFile:
        with open('test_regrLabels.txt', 'w') as regrFile:
            with open(os.path.join(rootDir, 'test_batch'), 'rb') as f:
                if sys.version_info[0] < 3: 
                    data = cp.load(f)
                else: 
                    data = cp.load(f, encoding='latin1')
                for i in range(10000):
                    fname = os.path.join(os.path.abspath(testDir), ('%05d.png' % i))
                    ut.saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 0)
