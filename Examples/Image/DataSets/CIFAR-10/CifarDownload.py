from __future__ import print_function
import sys
import cifar_utils as ut

def usage():
    print ('Usage: CifarDownload.py [-f <format>] \n  where format can be either cudnn or legacy. Default is cudnn.')

if __name__ == "__main__":
    fmt = ut.parseCmdOpt(sys.argv[1:])
    trn, tst = ut.loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', fmt)
    print ('Writing train text file...')
    ut.savetxt(r'./Train_cntk_text.txt', trn)
    print ('Done.')
    print ('Writing test text file...')
    ut.savetxt(r'./Test_cntk_text.txt', tst)
    print ('Done.')
