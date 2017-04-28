import h5py
import glob,os,sys,time
import numpy as np

from DLTools.ThreadedGenerator import  DLMultiClassGenerator,DLMultiClassFilterGenerator


def NEXTDataGenerator(Directory="/data/NEXT", batchsize=16, datasets=[u'3DImages'],**kwargs):

    Samples = [ (Directory+"/dnn_NEXT100_0vbb_si_v2x2x2_r200x200x200.Tensor.h5", datasets, "signal"    ),
                (Directory+"/dnn_NEXT100_Bi214_bg_v2x2x2_r200x200x200.Tensor.h5", datasets, "background"    )]

    def filterfunction(batchdict):
        r= np.array(range(batchdict["3DImages"].shape[0]))
        return r[0]

    
    GC= DLMultiClassFilterGenerator(Samples, batchsize=batchsize, FilterFunc=False,
                                    OneHot=True, shapes = [(batchsize, 200,200,200), (batchsize, 2)],  **kwargs)
    return GC
                     
if __name__ == '__main__':
    import sys
    Directory="/data/NEXT/tracks/"

    try:
        n_threads=int(sys.argv[1])
    except:
        n_threads=6

    try:
        n_threads2=int(sys.argv[2])
    except:
        n_threads2=n_threads

    Train_gen=NEXTDataGenerator(Directory,n_threads=n_threads,max=100000, verbose=False)
    
    print "Generator Ready"
    print "ClassIndex:", Train_gen.ClassIndexMap
    print "Object Shape:",Train_gen.shapes
    sys.stdout.flush()
    
    N=1
    NN=n_threads
    count=0
    old=start=time.time()
    for tries in xrange(1):
        print "*********************Try:",tries
        #for D in Train_gen.Generator():
        for D in Train_gen.Generator():
            NN-=0
            if NN<0:
                break
            start1=time.time()
            Delta=(start1-start)
            Delta2=(start1-old)
            old=start1
            print count,":",Delta, ":",Delta/float(N), Delta2
            sys.stdout.flush()
            N+=1
            for d in D:
                print d.shape
                #print d[np.where(d!=0.)]
                NN=d.shape[0]
                #print d[0]
                pass
            count+=NN
