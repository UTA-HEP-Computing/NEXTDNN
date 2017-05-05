import sys,os,argparse

# Parse the Arguments
execfile("NEXTDNN/ClassificationArguments.py")

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Use "--Test" to run on less events and epochs.
OutputBase="TrainedModels"
if TestMode:
    MaxEvents=int(10e3)
    NTestSamples=int(10e2)
    Epochs=10
    OutputBase+=".Test"
    print "Test Mode: Set MaxEvents to",MaxEvents,"and Epochs to", Epochs

if LowMemMode:
    n_threads=1
    multiplier=1
    
# Calculate how many events will be used for training/validation.
NSamples=MaxEvents-NTestSamples

# Function to help manage optional configurations. Checks and returns
# if an object is in current scope. Return default value if not.
def TestDefaultParam(Config):
    def TestParamPrime(param,default=False):
        if param in Config:
            return eval(param)
        else:
            return default
    return TestParamPrime

TestDefaultParam=TestDefaultParam(dir())

# Load the Data
from NEXTDNN.LoadData import * 

Train_genC=NEXTDataGenerator(InputDirectory,n_threads=n_threads,max=NSamples, bins=bins,verbose=False)
Test_genC=NEXTDataGenerator(InputDirectory,n_threads=n_threads,skip=NSamples, bins=bins,max=MaxEvents, verbose=False)
    
Train_gen=Train_genC.Generator()
Test_gen=Test_genC.Generator()

if Preload:
    print "Keeping data in memory after first Epoch. Hope you have a lot of memory."
    Train_gen=Train_genC.PreloadGenerator()
    Test_gen=Test_genC.PreloadGenerator()
    
# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from NEXTDNN.Models import *

# You can automatically load the latest previous training of this model.
if TestDefaultParam("LoadPreviousModel") and not LoadModel and BuildModel:
    print "Looking for Previous Model to load."
    ModelName=Name
    MyModel=ModelWrapper(Name=ModelName, LoadPrevious=True,OutputBase=OutputBase)

# You can load a previous model using "-L" option with the model directory.
if LoadModel and BuildModel:    
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/": LoadModel=LoadModel[:-1]
    MyModel=ModelWrapper(Name=os.path.basename(LoadModel),InDir=os.path.dirname(LoadModel),
                         OutputBase=OutputBase)
    MyModel.Load(LoadModel)

if BuildModel and not MyModel.Model:
    FailedLoad=True
else:
    FailedLoad=False

Shape=(None,)+bins
    
# Or Build the model from scratch
if BuildModel and not MyModel.Model :
    import keras
    print "Building Model...",

    MyModel=Fully3DImageClassification(Name, Shape, Width, Depth,
                                             BatchSize, NClasses,
                                             init=TestDefaultParam("WeightInitialization",'normal'),
                                             activation=TestDefaultParam("activation","relu"),
                                             Dropout=TestDefaultParam("DropoutLayers",0.5),
                                             BatchNormalization=TestDefaultParam("BatchNormLayers",False),
                                             NoClassificationLayer=False,
                                             OutputBase=OutputBase)

    # Configure the Optimizer, using optimizer configuration parameter.
    MyModel.Loss=loss
    # Build it
    MyModel.Build()
    print " Done."

if BuildModel:
    print "Output Directory:",MyModel.OutDir
    # Store the Configuration Dictionary
    MyModel.MetaData["Configuration"]=Config
    if "HyperParamSet" in dir():
        MyModel.MetaData["HyperParamSet"]=HyperParamSet

    # Print out the Model Summary
    MyModel.Model.summary()

    # Compile The Model
    print "Compiling Model."
    MyModel.BuildOptimizer(optimizer,Config)
    MyModel.Compile(Metrics=["accuracy"]) 

# Train
if Train or (RecoverMode and FailedLoad):
    print "Training."
    # Setup Callbacks
    # These are all optional.
    from DLTools.CallBacks import TimeStopping, GracefulExit
    from keras.callbacks import *
    callbacks=[ ]

    # Still testing this...

    if TestDefaultParam("UseGracefulExit",0):
        print "Adding GracefulExit Callback."
        callbacks.append( GracefulExit() )

    if TestDefaultParam("ModelCheckpoint",False):
        MyModel.MakeOutputDir()
        callbacks.append(ModelCheckpoint(MyModel.OutDir+"/Checkpoint.Weights.h5",
                                         monitor=TestDefaultParam("monitor","val_loss"), 
                                         save_best_only=TestDefaultParam("ModelCheckpoint_save_best_only"),
                                         save_weights_only=TestDefaultParam("ModelCheckpoint_save_weights_only"),
                                         mode=TestDefaultParam("ModelCheckpoint_mode","auto"),
                                         period=TestDefaultParam("ModelCheckpoint_period",1),
                                         verbose=0))

    if TestDefaultParam("EarlyStopping"):
        callbacks.append(keras.callbacks.EarlyStopping(monitor=TestDefaultParam("monitor","val_loss"), 
                                                       min_delta=TestDefaultParam("EarlyStopping_min_delta",0.01),
                                                       patience=TestDefaultParam("EarlyStopping_patience"),
                                                       mode=TestDefaultParam("EarlyStopping_mode",'auto'),
                                                       verbose=0))


    if TestDefaultParam("RunningTime"):
        print "Setting Runningtime to",RunningTime,"."
        TSCB=TimeStopping(TestDefaultParam("RunningTime",3600*6),verbose=False)
        callbacks.append(TSCB)
    

    # Don't fill the log files with progress bar.
    if sys.flags.interactive:
        verbose=1
    else:
        verbose=1 # Set to 2

    print "Evaluating score on test sample..."
    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)
    
    print "Initial Score:", score
    MyModel.MetaData["InitialScore"]=score
        
    MyModel.History = MyModel.Model.fit_generator(Train_gen,
                                                  steps_per_epoch=(NSamples/BatchSize),
                                                  epochs=Epochs,
                                                  verbose=verbose, 
                                                  validation_data=Test_gen,
                                                  validation_steps=NTestSamples/BatchSize,
                                                  callbacks=callbacks)

    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)


    print "Evaluating score on test sample..."
    print "Final Score:", score
    MyModel.MetaData["FinalScore"]=score

    if TestDefaultParam("RunningTime"):
        MyModel.MetaData["EpochTime"]=TSCB.history

    # Store the parameters used for scanning for easier tables later:
    for k in Params:
        MyModel.MetaData[k]=Config[k]

    # Save Model
    MyModel.Save()
else:
    print "Skipping Training."
    
# Analysis
if Analyze:
    print "Running Analysis."
    # Data is too big to store in memory... will load a small fraction.
    # Should write Analysis that uses batched data
    Test_genC=NEXTDataGenerator(InputDirectory,n_threads=n_threads,skip=NSamples,
                                bins=bins,
                                max=BatchSize*4, verbose=False)

    Test_genC.PreloadData()
    Test_X, Test_Y = tuple(Test_genC.D)

    from DLAnalysis.Classification import MultiClassificationAnalysis
    result,NewMetaData=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize,PDFFileName="ROC",
                                                   IndexMap=Test_genC.ClassIndexMap)

    MyModel.MetaData.update(NewMetaData)
    
    # Save again, in case Analysis put anything into the Model MetaData
    if not sys.flags.interactive:
        MyModel.Save()
    else:
        print "Warning: Interactive Mode. Use MyModel.Save() to save Analysis Results."
        
# Make sure all of the Generators processes and threads are dead.
# Not necessary... but ensures a graceful exit.
# if not sys.flags.interactive:
#     for g in GeneratorClasses:
#         try:
#             g.StopFiller()
#             g.StopWorkers()
#         except:
#             pass
