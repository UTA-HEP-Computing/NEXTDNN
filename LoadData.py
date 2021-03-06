import h5py
import glob,os,sys,time
import numpy as np

from DLTools.ThreadedGenerator import  DLMultiClassGenerator,DLMultiClassFilterGenerator


def NEXTDataGeneratorOld(Directory="/data/NEXT", batchsize=16, datasets=[u'3DImages'],**kwargs):

    Samples = [ (Directory+"/dnn_NEXT100_0vbb_si_v2x2x2_r200x200x200.Tensor.h5", datasets, "signal"    ),
                (Directory+"/dnn_NEXT100_Bi214_bg_v2x2x2_r200x200x200.Tensor.h5", datasets, "background"    )]

    def filterfunction(batchdict):
        r= np.array(range(batchdict["3DImages"].shape[0]))
        return r[0]

    
    GC= DLMultiClassFilterGenerator(Samples, batchsize=batchsize, FilterFunc=False,
                                    OneHot=True, shapes = [(batchsize, 200,200,200), (batchsize, 2)],  **kwargs)
    return GC


def NEXTDataGenerator(Directory="/data/NEXT", batchsize=16, datasets=['Hits/C','Hits/V'], Norm=True,
                      bins=(200,200,200),**kwargs):

    Samples = [ (Directory+"/dnn_NEXT100_0vbb_si_v2x2x2_r200x200x200.Tensor.h5", datasets, "signal"    ),
                (Directory+"/dnn_NEXT100_Bi214_bg_v10x10x5_r200x200x200.Tensor.h5", datasets, "background"    )]

    def MakeImage(bins,Norm=True):
        def f(D):
            for i in xrange(D[0].shape[0]):
                if Norm:
                    w=np.tanh(np.sign(D[1][i]) * np.log(np.abs(D[1][i]) + 1.0) / 2.0)
                else:
                    w=D[1][i]
                R,b=np.histogramdd(D[0][i], bins=list(bins), weights=w)
            return [R]+D[2:]
        return f

    
    GC= DLMultiClassGenerator(Samples, batchsize=batchsize,
                              preprocessfunction=MakeImage(bins,Norm),
                              OneHot=True,
                              shapes = [(batchsize,)+bins,(batchsize, 2)],
                              **kwargs)
    return GC


Test=1
                     
if __name__ == '__main__' and Test==0:
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

    Train_gen=NEXTDataGeneratorOld(Directory,n_threads=n_threads,max=100000, verbose=False)
    
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


if __name__ == '__main__' and Test==1:
    import sys
    Directory="/data/NEXT/tracksVL/"

    try:
        n_threads=int(sys.argv[1])
    except:
        n_threads=6

    try:
        n_threads2=int(sys.argv[2])
    except:
        n_threads2=n_threads

    Train_gen=NEXTDataGenerator(Directory,n_threads=n_threads,max=100000,
                                bins=(100,100,100),verbose=False)
    
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
                print d[np.where(d!=0.)]
                NN=d.shape[0]
                #print d[0]
                pass
            count+=NN
