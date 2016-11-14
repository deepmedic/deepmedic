The DeepMedic
=====================================

### News
14 Nov 2016 (v0.5.4):
* Original configuration moved to deepMedicOriginal. Updated config now in deepMedic.
* More memory efficient testing. CNN code has been refactored. Minor fixes.

10 Oct 2016 (v0.5.3):
* Sampling refactored. Now possible to use weighted-maps to sample each class.

4 Aug 2016 (v0.5.2):
* Code in Layer-classes cleaned/commented. Residual Connections enabled. Possible to specify kernel size at FC1. 

14 July 2016 (v0.5.1):
* Master branch was updated with better monitoring of training progress and a better plotting script. This version is not backwards compatible. CPickle will fail loading previously trained models from previous versions of the code.
* Previous version of master branch tagged as v0.5. Use this if you wish to continue working with previously trained models.

Important Issue:
* Current saving of CNN's state is not backwards compatible. This means that any change to the code of the cnn3d.py and cnnLayerTypes.py will not allow CPickle to load models created with previous versions. This has big priority to solve. Until then, please use tags (versions) compatible with your trained models.

### Introduction

This project aims to offer easy access to Deep Learning for segmentation of structures of interest in biomedical 3D scans. It is a system that allows the easy creation of a 3D Convolutional Neural Network, which can be trained to detect and segment structures if corresponding ground truth labels are provided for training. The system processes NIFTI images, making its use straightforward for many biomedical tasks.

This document describes how to install and run the software. Accompanying data are provided to run the preset examples and make sure that the system is functioning on your system. This document also describes the main functionality, in order for the user to understand the main processing cycle. For greater details please consult [1]. We hope this project will serve well in making the state-of-the-art Convolutional Networks more accessible in the field of medical imaging.

#### Citations

The system was initially developed for the segmentation of brain lesions in MRI scans. It was employed for our research presented in [1],[2], where a 3D network architecture with two convolutional pathways was presented for the efficient multi-scale processing of multi-modal MRI volumes. If the use of the software positively influences your endeavours, please cite [1].

[1] **Konstantinos Kamnitsas**, Christian Ledig, Virginia F.J. Newcombe, Joanna P. Simpson, Andrew D. Kane, David K. Menon, Daniel Rueckert, and Ben Glocker, “[Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation][paper1]”, *Medical Image Analysis, 2016*.

[2] **Konstantinos Kamnitsas**, Liang Chen, Christian Ledig, Daniel Rueckert, and Ben Glocker, “[Multi-Scale 3D CNNs for segmentation of brain Lesions in multi-modal MRI][paper2]”, *in proceeding of ISLES challenge, MICCAI 2015*.


### Table of Contents
* [1. Installation and Requirements](#1-installation-and-requirements)
  * [1.1. Required Libraries](#11-required-libraries)
  * [1.2. Installation](#12-installation)
  * [1.3. Required Data Pre-Processing](#13-required-data-pre-processing)
  * [1.4. GPU Processing](#14-gpu-processing)
* [2. Running the Software](#2-running-the-software)
  * [2.1 Training a tiny CNN - Making sure it works](#21-training-a-tiny-cnn---making-sure-it-works)
  * [2.2 Common errors when utilizing a GPU](#22-common-errors-when-utilizing-a-gpu)
* [3. How it works](#3-how-it-works)
  * [3.1 Model Creation](#31-model-creation)
  * [3.2 Training](#32-training)
  * [3.3 Testing](#33-testing)
* [4. How to run DeepMedic on your data](#4-how-to-run-deepmedic-on-your-data)
* [5. Concluding](#5-concluding)
* [6. Licenses](#6-licenses)

### 1. Installation and Requirements

#### 1.1. Required Libraries

The system is written in python. The following libraries are required:

- [Theano](http://deeplearning.net/software/theano/): This is the Deep Learning library that the back end is implemented with.
- [Nose](https://pypi.python.org/pypi/nose/): Required for Theano’s unit tests.
- [NiBabel](http://nipy.org/nibabel/): The library used for loading NIFTI files.
- [Parallel Python](http://www.parallelpython.com/): Library used to parallelize parts of the training process.
- [six](https://pypi.python.org/pypi/six) : Python compatibility library.
- [scipy](https://www.scipy.org/) : Package of tools for statistics, optimization, integration, algebra, machine learning.
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
 

#### 1.2. Installation

The software can be found at `https://github.com/Kamnitsask/deepmedic/`. After cloning the project, all the dependencies can be installed by running the following command in the root directory:

```python
python setup.py install
```

This will download all required libraries, install and add them to the environment's PATH. This should be enough to use the DeepMedic.

*Alternatively*, the user can manually add them to the PATH, or use the provided **environment.txt** file:

```python
#=============== LIBRARIES ====================
#Theano is the main deep-learning library used. Version >= 0.8 required. Link: http://deeplearning.net/software/theano/
path_to_theano = '/path/to/theano/on/the/filesystem/Theano/'

#Nose is needed by Theano for its unit tests. Link: https://pypi.python.org/pypi/nose/
path_to_nose = '/path/to/nose/on/the/filesystem/nose_installation'

#NiBabel is used for loading/saving NIFTIs. Link: http://nipy.org/nibabel/
path_to_nibabel = '/path/to/nibabel/on/the/filesystem/nibabel'

#Parallel-Python is required, as we extract training samples in parallel with gpu training. Link: http://www.parallelpython.com/
path_to_parallelPython = '/path/to/pp/on/the/filesystem/ppBuild'
```

 The latter file is parsed by the main software. If the lines with the corresponding lines are **not** commented out, the given path will be internally pre-pended in the PATH.

#### 1.3. Required Data Pre-Processing

* DeepMedic processes **NIFTI files** only. All data should be in the *.nii* format.

* The input modalities, ground-truth labels, ROI masks and other **images of each subject need to be co-registered** (per-subject, no need for inter-subject registration). 

* The images of each subject should **have the same dimensions** (per subject, no need for whole database). This is, the number of voxels per dimension must be the same for all images of a subject. 

* **Resample all images in the database to the same voxel size**. The latter is needed because the kernels (filters) of the DeepMedic need to correspond to the same real-size patterns (structures) **for all subjects**.

* Make sure that the **ground-truth labels** for training and evaluation represent the background with zero. The system also assumes that the task’s classes are indexed increasing by one (not 0,10,20 but 0,1,2).

* **You are strongly advised to normalize the intensity of the data within the ROI to a zero-mean, unary-variance space**. Our default configuration significantly underperforms if intensities are in another range of values.

**Note for large images**: Large 3D CNNs are computationally expensive. Consider downsampling the images or reducing the size of the network if you encounter computational difficulties. The default configuration of DeepMedic was applied on scans of size around 200x200x200. 

#### 1.4. GPU Processing

Small networks can be run on the cpu. But 3D CNNs of considerable size require processing on the GPU. For this, an installation of [Nvidia’s CUDA](https://developer.nvidia.com/cuda-toolkit) is needed. Make sure to acquire a version compatible with your GPU drivers. Theano needs to be able to find CUDA’s compiler, the **nvcc**, in the environment’s path. It also dynamically links to **cublas.so** libraries, which need to be visible in the environment’s.

Prior to running DeepMedic on the GPU, you must manually add the paths to the folders containing these files in your environment's variables. As an example, in a *cshell* this can be done with *setenv*:

```cshell
setenv PATH '/path-to/cuda/7.0.28/bin':$PATH
setenv LD_LIBRARY_PATH '/path-to/cuda/7.0.28/lib64'
```





### 2. Running the Software

The source code of the DeepMedic is provided in the folder [deepmedic](deepmedic/). Users should not need to touch this folder. The software comes with a command line interface, [deepMedicRun](deepMedicRun). Running it with the help option:
```cshell
./deepMedicRun -h
```
brings up the available actions for the creation, training and testing of CNN models. All actions require a large number of configuration parameters, which are read from configuration files. 

In the [examples/configFiles](examples/configFiles/) folder we provide two sets of configuration files. Firstly, the configuration of a very small network is given in [examples/configFiles/tinyCnn/](examples/configFiles/tinyCnn/). This network can be trained within minutes on a CPU. It's a simple example [to make sure everything works](#21-training-a-tiny-cnn---making-sure-it-works). We also provide the full configuration of the DeepMedic model, as employed in [[1](#citations)], in the folder [examples/configFiles/deepMedicBrats/](examples/configFiles/deepMedicBrats/).

The above configuration files are pre-set to point to accompanying .nii files, provided in [examples/dataForExamples/](examples/dataForExamples/). Those NIFTIs serve as input to the networks in our examples. This data are modified versions of images from the Brain Tumor Segmentation challenge ([BRATS 2015](http://braintumorsegmentation.org/)).


#### 2.1 Training a tiny CNN - Making sure it works

We will here train a tiny CNN model and make sure everything works as expected. Further explanations on the use of the software are provided in the next section.

NOTE: First see [Section 1.2](#12-installation) for installation of the required packages. 

Lets **create** a model :
```cshell
./deepMedicRun -newModel ./examples/configFiles/tinyCnn/model/modelConfig.cfg
```

This command parses the given configuration file, creates a CNN model with the specified architecture, initializes and saves it. The folder `./examples/output/` should have been created by the process, where all output is saved. When the process finishes (roughly after a couple of minutes) a new and untrained model should be saved using [cPickle](https://docs.python.org/2/library/pickle.html) at `./examples/output/cnnModels/tinyCnn.initial.DATE+TIME.save`. All output of the process is logged for later reference. This should be found at `examples/output/logs/tinyCnn.txt`. Please make sure that the process finishes normally, the model and the logs created. If everything looks fine, briefly rejoice and continue... 

Lets **train** the model with the command (replace *DATE+TIME*):
```cshell
./deepMedicRun -train examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg \
                       -model examples/output/cnnModels/tinyCnn.initial.DATE+TIME.save
```

The model should be loaded and training for two epochs should be performed. After each epoch the trained model is saved at `examples/output/cnnModels/trainSessionWithValidTinyCnn`. The logs for the sessions should be found in `examples/output/logs/trainSessionWithValidTinyCnn.txt`. Finally, after each epoch, the model performs segmentation of the validation images and the segmentation results (.nii files) should appear in `examples/output/predictions/trainSessionWithValidTinyCnn/predictions/`. If the training finishes normally (should take 5 mins) and you can see the mentioned files in the corresponding folders, beautiful. 

You can **plot the training progress** using an accompanying script, which parses the training logs:
'''
python plotTrainingProgress.py examples/output/logs/trainSessionWithValidTinyCnn.txt -d
'''

Now lets **test** with the trained model (replace *DATE+TIME*):
```cshell
./deepMedicRun -test examples/configFiles/tinyCnn/test/testConfig.cfg \
                       -model examples/output/cnnModels/trainSessionWithValidTinyCnn/tinyCnn.trainSessionWithValidTinyCnn.final.DATE+TIME.save
```

This should perform segmentation of the testing images and the results should appear in `examples/output/predictions/testSessionTinyCnn/` in the `output` folder. In the `features` folder you should also find some files, which are feature maps from the second layer. DeepMedic gives you this functionality (see testConfig.cfg). If the testing process finishes normally and all output files seem to be there, **everything seems to be working!** *On the CPU*... 

Now lets check the important part... If using the **DeepMedic on the GPU** is alright on your system. First, delete the `examples/output/` folder for a clean start. Now, most importantly, place the path to **CUDA**'s *nvcc* into your *PATH* and to the *cublas.so* in your *LD_LIBRARY_PATH* (see [section 1.4](#14-gpu-processing))

You need to perform the steps we did before for creating a model, training it and testing with it, but on the GPU. To do this, repeat the previous commands and pass the additional option `-dev gpu`. For example: 

```cshell
./deepMedicRun -dev gpu -newModel ./examples/configFiles/tinyCnn/model/modelConfig.cfg
```
The processes should result in similar outputs as before. If all processes finish as normal and you get no errors, amazing. **Now it seems that really everything works :)** Continue to the next section and find more details about the DeepMedic and how to use the large version of our network!

#### 2.2 Common errors when utilizing a GPU

Common errors that indicate something is wrong with the CUDA installation or your environment variables :
```
EnvironmentError: You forced the use of gpu device 'gpu', but nvcc was not found. Set it in your PATH environment variable or set the Theano flags 'cuda.root' to its directory
```
The above may be thrown because the CUDA nvcc compiler is not correctly set in your PATH.

```
ERROR (theano.sandbox.cuda): Failed to compile cuda_ndarray.cu: libcublas.so.6.5: cannot open shared object file: No such file or directory
...
EnvironmentError: You forced the use of gpu device gpu, but CUDA initialization failed with error:
cuda unavailable
```
The above is probably thrown because the CUDA libraries such as cublas.so are not correctly set in your LD_LIBRARY_PATH.

```
EnvironmentError: You forced the use of gpu device gpu, but CUDA initialization failed with error:
Unable to get the number of gpus available: CUDA driver version is insufficient for CUDA runtime version
```
or something like:
```
Exception: ('The following error happened while compiling the node', GpuCAReduce{add}{1}(<CudaNdarrayType(float32, vector)>), '\n', 'nvcc return status', 1, 'for cmd', 'nvcc -shared -O3 -arch=sm_52 -m64 
-Xcompiler -fno-math-errno,-Wno-unused-label,-Wno-unused-variable,-Wno-write-strings,-DCUDA_NDARRAY_CUH=c72d035fdf91890f3b36710688069b2e,-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION,-fPIC,-fvisibility=hidden 
...
-L/usr/lib -lcudart -lcublas -lcuda_ndarray -lpython2.7', '[GpuCAReduce{add}{1}(<CudaNdarrayType(float32, vector)>)]')
```
may be thrown because you are using a version of CUDA that is not compatible with your GPU's drivers. *Note: On March 2016, I ve been using v7.0 for NVIDIA Titans, K40s, GTX 980s, and v6.5 for anything older.*


### 3. How it works

Previously we briefly discussed how to quickly run a pre-set example with a tiny CNN, just so you can check whether everything works on your system. In this section we will go through the process in a bit more detail. We also explain the main parameters that should be specified in the configuration files, in order for you to tailor the network and process to your needs. 

The **.cfg configuration files** in `examples/configFiles/deepMedicOriginal/` display the parameters used in our work in [[1](#citations)]. In an attempt to make it simpler for the user, we also provide a cleaner version of the configuration files, named with "Less", where many parameters are "hidden". They are internally passed values that worked well in our experiments. Finally, the config files in `examples/configFiles/deepMedic/` provide a network configuration that we will be gradually updating with components that seem to improve the overall performance of the system.

**_Note:_** The config files are parsed as python scripts, thus follow **python syntax**. Any commented-out configuration variables are internally given **default values**.

#### 3.1. Model Creation

To create a new CNN model, you need to point to a config file with the parameters of a model:
```
./deepMedicRun -dev gpu -newModel ./examples/configFiles/deepMedic/model/modelConfig.cfg
```

After reading the parameters given in modelConfig.cfg, a CNN-model will be created and saved with cPickle in the output folder. The session prints all the parameters that are used for the model-creation on the screen and to a log.txt file. 

**Parameters for defining Network Architecture**

![alt text](documentation/deepMedic.png "Double pathway CNN for multi-scale processing")
Figure 1: The architecture of an example of our double-pathway architecture for multi-scale processing. At each layer, the number and size of feature maps (FMs) is depicted in the format (*Number-Of-FMs x Dimensions*).

The main parameters to specify the CNN model are the following.

*Generic:*

- modelName: the cnn-model’s name, will be used for naming the files that the model is being saved with after its creation, but also during training.
- folderForOutput: The main output folder. Saved model and logs will be placed here.

*Task Specific:*

- numberOfOutputClasses: DeepMedic is multiclass system. This number should include the background, and defines the number of FMs in the last, classification layer (=2 in Fig.1)
- numberOfInputChannels: Specify the number of modalities/sequences/channels of the scans.

*Architecture:*

- numberFMsPerLayerNormal: A list which needs to have as many entries as the number of layers in the normal pathway that  we want to create. Each entry is a number, which defines the number of feature-maps in the corresponding layer ([30, 40, 40, 50] in fig1)
- kernelDimPerLayerNormal: The dimensions of the kernels per layer. ([[5,5,5], [5,5,5], [5,5,5], [5,5,5]] in Fig.1.) 
- useSubsampledPathway: Setting this to “True” creates a subsampled-pathway, with the same architecture as the normal one. “False” for single-scale processing with the normal pathway only. Additional parameters allow tailoring this pathway further.
- numberFMsPerLayerFC: The final layers of the two pathways are contatenated. This parameter allows the addition of extra hidden FC layers before the classification layer. The number of entries specified how many extra layers, the number of each entry specifies the number of FMs in each layer. Final classification layer not included ([[150], [150]] in Fig.1).

*Image Segments and Batch Sizes:*

- segmentsDim(Train/Val/Inference): The dimensions of the input-segment. Different sizes can be used for training, validation, inference (testing). Bigger sizes require more memory and computation. Training segment size greatly influences distribution of training samples ([25,25,25] in Fig.1). Validation segments are by default as large as the receptive field (one patch). Size of testing-segments only influences speed.
- Batch Size : The number of segments to process simultaneously on GPU. In training, bigger batch sizes achieve better convergence and results, but require more computation and memory. Batch sizes for Validation and Inference are less important, greater once just speedup the process.

More variables are available, but are of less importance (regularization, optimizer, etc). They are described in the config files of the provided examples. 

#### 3.2. Training

After a model is created, it is pickled and saved. You can then train it using your manually segmented, ground-truth annotations. Again, you need to point to a configuration file with the parameters of the training session. Training can be started in 3 ways.

a) By adding the `-model` option and specifying the file with a saved/pickled model (created by a -newModel process or half trained from a previous training-session).
```
./deepMedicRun -dev gpu -train ./examples/configFiles/deepMedic/train/trainConfig.cfg \
                                -model ./path-to-saved-model
```

b) The path to the saved model to train can be specified in the config file. In this case, the `-model` option can be ommited. **Note:** A file specified by `-model` option overrides any specified in the config-file.
```
./deepMedicRun -dev gpu -train ./examples/configFiles/deepMedic/train/trainConfig.cfg
```

c) The model created from a model-creation session can be passed straight to a training session, after it is saved. **Note:** If a path to another model is specified in the config-file, it is disregarded.
```
./deepMedicRun -dev gpu -newModel ./examples/configFiles/deepMedic/model/modelConfig.cfg \
                       		-train ./examples/configFiles/deepMedic/train/trainConfig.cfg
```

**The Training Session**

During a training session, the following cycle is followed:
```
For each epoch {
	For each subepoch {
		Load Validation Cases
		Extract Validation Segments
		Perform Validation-On-Samples (in batches)

		Load Training Cases
		Extract Training Segments
		Perform Training (in batches)

		Report Accuracy over subepoch’s samples (Val/Train).
	}
	Report Accuracy over whole epoch (Val/Train)
	Lower Learning Rate if required.
	Save the model’s state

	Perform Full-Inference on Validation Cases (every few epochs)
	Report DSC of full inference
	Save predictions from Full-Inference (segm./prob.maps/features)
}
```

The validation on samples and the full segmention of the scans of validation subjects are optional. The **progress of training can be plotted** by using the accompanying `plotTrainingProgress.py` script, which parses the training logs for the reported validation and training accuracy metrics:
```
python plotTrainingProgress.py examples/output/logs/trainSessionDeepMedic.txt -d
```

**Training Parameters**

*Generic Parameters:*

- sessionName: The name of the session. Used to save the trained models, logs and results.
- folderForOutput: The main output folder.
- cnnModelFilePath: path to a saved CNN model (disregarded if training started as b or c).

*Input for Training:*

- channelsTraining: For each of the input channels, this list should hold one entry. Each entry should be a path to a file. These files should list the paths to the corresponding channels for each of the training subjects. See the pre-set files given in the examples and all these should easily become clear.
- gtLabelsTraining: the path to a file. That file should list the paths to the ground-truth labels for all training subjects.
- roiMasksTraining: In many tasks we can easily define a Region Of Interest and get a mask of it. For instance by excluding the air in a body scan, or take the brain-mask in a brain scan. In this case, this parameter allows pointing to the roi-masks for each training subject. Sampling or inference will not be performed outside this area, focusing the learning capacity of the network inside it. If this is not available, detete or comment this variable out and sampling will be performed on whole volumes.

*Training Cycle:*

- numberOfEpochs: Total number of epochs until the training finishes.
- numberOfSubepochs: Number of subepochs to run per epoch
- numOfCasesLoadedPerSubepoch: At each subepoch, the images from maximum that many cases are loaded to extract training samples. This is done to allow training on databases that may have hundreds or thousands of images, and loading them all for sample-extraction would be just too expensive.
- numberTrainingSegmentsLoadedOnGpuPerSubep: At every subepoch, we extract in total this many segments, which are loaded on the GPU in order to perform the optimization steps. Number of optimization steps per subepoch is this number divided by the batch-size-training (see model-config). The more segments, the more GPU memory and computation required.

*Learning Rate Schedule:*

- stable0orAuto1orPredefined2orExponential3LrSchedule : Schedules to lower the Learning Rate with. Stable lowers LR every few epochs. Auto lowers it when validation accuracy plateaus (unstable). Predefined requires the user to specify which epochs to lower it. Exponential lowers it over time while it increases momentum. We advice to use constant LR, observe progress of training and lower it manually when improvement plateaus. Otherwise, use exponential, but make sure that training is long enough to ensure convergence before LR is significantly reduced.

*Data Augmentation:*

- reflectImagesPerAxis: Specify whether you d like the images to be randomly reflected in respect to each axis, for augmentation during training.
- performIntAugm: Randomly apply a change to segments’ intensities: I' = (I + shift) * multi

*Validation:*

- performValidationOnSamplesThroughoutTraining, performFullInferenceOnValidationImagesEveryFewEpochs: Booleans to specify whether we want to perform validation, since it is actually time consuming.
- channelsValidation, gtLabelsValidation, roiMasksValidation: Similar to the corresponding training entries. If default settings for validation-sampling are enabled, sampling for validation is done in a uniform way over the whole volume, to achieve correct distribution of the classes.
- numberValidationSegmentsLoadedOnGpuPerSubep: on how many validation segments (samples) to perform the validation.
- numberOfEpochsBetweenFullInferenceOnValImages: Every how many epochs to perform full-inference validation. It might be slow to process all validation cases often.
- namesForPredictionsPerCaseVal: If full inference is performed, we may as well save the results to visually check progress. Here you need to specify the path to a file. That file should contain a list of names, one for each case, with which to save the results. Simply the names, not paths. Results will be saved in the output folder.

#### 3.3. Testing

When a training epoch is finished, the model’s state is saved. These pickled models can be used for segmenting previously unseen scans. A testing configuration file has to be specified. Testing can be started in two ways.


a) A model is specified straight from the command line.
```
./deepMedicRun -dev gpu -test ./examples/configFiles/deepMedicBratsClean/ test/testConfig.cfg \
                       		-model ./path-to-saved-model
```

b) The path to a saved model can be instead specified in the testing config file, and then the `-model` option can be ommited. **Note:** A file specified by `-model` option overrides any specified in the config-file.
```
./deepMedicRun -dev gpu -test ./examples/configFiles/deepMedicBratsClean/test/testConfig.cfg
```

After the model is loaded, inference will be performed on the testing subjects. Predicted segmentation masks, posterior probability maps for each class,  as well as the feature maps of any layer can be saved. If ground-truth is provided, DeepMedic will also report DSC metrics for its predictions.

Note that this testing procedure is similar to the full-inference procedure performed on validation subjects every few training epochs.

**Testing Parameters**

*Main Parameters:*

- sessionName: The name for the session, to use for saving the logs and inference results.
- folderForOutput: The output folder to save logs and results.
- cnnModelFilePath: The path to the cnn model to use. Disregarded if specified from command line.
- channels: List of paths to the files that list the files of channels per testing case. Similar to the corresponding parameter for training.
- namesForPredictionsPerCase: Path to a file that lists the names to use for saving the prediction for each subject.
- roiMasks: If masks for a restricted Region-Of-Interest can be made, inference will only be performed within it. If this parameter is omitted in the config file, whole volume is scanned.
- gtLabels: Path to a file that lists the file-paths to Ground Truth labels per case. Not required for testing, but if given, DSC accuracy metric is reported.

*Saving Predictions:*

- saveSegmentation, saveProbMapsForEachClass : Specify whether you would like the segmentation masks and the probability maps of a class saved.

*Saving Feature Maps:*

- saveIndividualFms, saveAllFmsIn4DimImage : Specify whether you would like the feature maps saved. Possible to save each FM in a separate files, or create a 4D file with all of them. Note that FMs are many and the 4D file can be several hundreds of MBs, or GBs.
- minMaxIndicesOfFmsToSaveFromEachLayerOfABCPathway : Because the number of FMs is large, it is possible to specify particular FMs to save. Provide the minimum (inclusive) and maximum (exclusive) index of the FMs of the layers that you would like to save (indexing starts from 0).


### 4. How to run DeepMedic on your data

In `examples/configFiles/deepMedicOriginal/` we provide the configuration of the network as employed in our work in [1], and very similar to the model employed in our winning contribution for the ISLES 2015 challenge [2]. The config files named with “Less” are “cleaner” versions to make them more readable, where many parameters are omitted/hidden (they are passed *default* values internally). The configuration of these two models is exactly the same. In `examples/configFiles/deepMedic/` we provide a configuration which we will be gradually updating with any components we find generally well behaved. You are adviced to use the latter, bearing in mind that behavior might slightly change between version (hopefully for the best!). 

To run the DeepMedic on your data, the following are the minimum steps you need to follow:

**a)** **Pre-process your data** as described in Sec. [1.3](#13-required-data-pre-processing). Do not forget to normalise them to a zero-mean, unary-variance space. Produce ROI masks (for instance brain masks) if possible for the task.

**b)** In the **modelConfig.cfg** file, change the variable `numberOfOutputClasses = 5` to the number of classes in your task (eg 2 if binary), and `numberOfInputChannels = 2` to the number of input modalities. Now you are ready to create the model via the `-newModel` option.

**c)** (optional) If you want to train a bigger or smaller network, the easiest way is to increase/decrease the number of Feature Maps per layer. This is done by changing the number of FMs in the variable `numberFMsPerLayerNormal = [30, 30, 40, 40, 40, 40, 50, 50]`.

**d)** Before you train a network you need to alter the **trainConfig.cfg** file, in order to let the software know where your input images are. The variable `channelsTraining = ["./trainChannels_flair.cfg", "./trainChannels_t1c.cfg"]` is pre-set to point to two files, one for each of the input variables. Adjust this for your task. 

**e)** Create your files that correspond to the above `./trainChannels_flair.cfg, trainChannels_t1c.cfg` files for your task. Each of these files is essentially a list. Every file has an entry for each of the training subjects. The entry is the path to the .nii file with the corresponding modality image for the subject. A brief look to the provided exemplary files should make things clear.

**f)** Do the same process in order to point to the ground-truth labels for training via the variable `gtLabelsTraining = "./trainGtLabels.cfg"` and to ROI masks (if available) via `roiMasksTraining = "./trainRoiMasks.cfg"`.

**g)** If you wish to periodically perform **validation** throughout training, similar to the above, point to the files of validation subjects via the variables `channelsValidation`, `gtLabelsValidation` and `roiMasksValidation`. If you do not wish to perform validation (it is time consuming), set to `False` the variables `performValidationOnSamplesThroughoutTraining`
and `performFullInferenceOnValidationImagesEveryFewEpochs`.

**h)** (optional) If you need to adjust the length of the training session, eg for a smaller network, easiest way is to lower the total number of epochs `numberOfEpochs=35`. You should also then adjust the pre-defined schedule via `predefinedSchedule`. Another option is to use exponentially decreasing schedule for the learning rate, by setting `stable0orAuto1orPredefined2orExponential3LrSchedule = 3`.

**i)** **To test** a trained network, you need to point to the images of the testing subjects, similar to point d) for the training. Adjust the variable `channels = ["./testChannels_flair.cfg", "./testChannels_t1c.cfg"]` to point to the modalities of the test subjects. If ROI masks are available, point to them via `roiMasks` and inference will only be performed within the ROI. Else comment this variable out. Similarly, if you provide the ground-truth labels for the testing subjects via `gtLabels`, accuracy of the prediction will be calculated and the DSC metric will be reported. Otherwise just comment this variable out.

**j)** Finally, you need to create a file, which will list names to give to the predictions for each of the testing subject. See entry `namesForPredictionsPerCase = "./testNamesOfPredictionsSimple.cfg"` and the corresponding pre-set file. After that, you are ready to test with a model.

The provided configuration of the DeepMedic takes roughly 2 days to get trained on an NVIDIA GTX Titan X. Inference on a standard size brain scan should take 2-3 minutes. Adjust configuration of training and testing or consider downsampling your data if it takes much longer for your task.

### 5. Concluding

We hope that making this software publically available can accelerate adoption of Deep Learning in the field of Biomedical Image Processing and Analysis. It is still actively developed, so do take into account that it is far from modular and fully generic. In its current state, we believe it can serve as a helpful **baseline method** for the benchmarking of further learners, as well as a segmentation system for various pipelines. If you find our work has positively influenced yours, please cite our paper [1].

I am well aware that much of the functionality is not fully modular and the API far from perfect. I hope to make DeepMedic a helpful tool for the community. So feel free to email me with your feedback or any issues at: **konstantinos.kamnitsas12@ic.ac.uk**

Best wishes,

Konstantinos Kamnitsas

### 6. Licenses

License for the DeepMedic software: BSD 3-Clause License. A copy of this license is present in the root directory.

License for the data provided for the examples: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Switzerland License. http://creativecommons.org/licenses/by-nc-sa/3.0/ch/deed.en


[//]: # (reference links)

   [paper1]: <http://www.sciencedirect.com/science/article/pii/S1361841516301839>

   [paper2]: <http://www.isles-challenge.org/ISLES2015/articles/kamnk1.pdf>
