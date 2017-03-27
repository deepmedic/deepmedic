
Pre-trained models, transfer weights or fine-tune
=====================================

A commonly used technique with neural networks is transfer of knowledge acquired from a database or task to another. A straightforward way to do this is via training a model on a database or task A first, and then use this knowledge for better learning from another database or task B. Below is described relevant functionality provided.

Disclaimer: This functionality has not been as thoroughly validated as the original segmentation system. If you find or suspect there is a bug, email me at konstantinos.kamnitsas12@imperial.ac.uk. Use with care.


### 1. Fine-tune a previously trained model with another training session

Lets assume that using this software you have created a model, named **modelA**, and it has been trained on a database named **dataA**, with a training session named **trainSessionA**. The trained model is now pickled in a file **modelA.trainSessionA.date.save**. 

You have a second database with available training labels, **dataB**, which is relevant to *dataA*. For instance, *dataA* may be brain MRI data of patients with stroke lesions and *dataB* may be brain MRI data from subjects with brain tumors. You wish to transfer the knowledge of the previously trained *modelA* to better learn to segment the tumors in *dataB*.

**If** both databases have the **same number of channels** and **number of classes**, for instance, both are binary segmentation, then the very same model that was previously trained, *modelA*, can be **fine-tuned** on dataB for the new task. 

A straight-forward way to do this is to simply continue training *modelA* in **a new training session**, *trainSessionB*, using *dataB*. For this, all you need is to create a new trainingSessionB configuration file, similar to whatever used for *trainSessionA*, which however is now pointing to *dataB*. **However there is a catch!**

When we save a model, besides all trainable parameters, we also save everything relevant to the optimization in order to preserve its whole state and allow seamless continuation of training, for instance for the case that training was interrupted (See Sec. 3.2 in [main documentation](https://github.com/Kamnitsask/deepmedic/blob/master/documentation/README.md)). Examples are the number of epochs previously trained, the learning rate schedule used and its current stage, velocity/momentum of the updates etc. In order to refine a previously trained model with a new training session, we should **reset it's optimization state** and reinitialize it with the new configuration. Otherwise, for instance if the model was already trained for 35 epochs, the optimizer at the beginning of training Session B will continue from epoch 35, with the previous learning rate schedule, which is probably not what we want.

To reset the optimizer and reinitialize it with the new configuration of trainining Session B, all you need is to provide the option `-resetOptimizer` in the command line when the new training session is initiated:
```cshell
python ./deepMedicRun -dev gpu -train ./path/to/trainConfigForSessionB.cfg -resetOptimizer \
                                   -model ./path/to/previously/trained/model/modelA.trainSessionA.date.save
```


The training session will begin like normal, but the optimizer will be reinitialized. To double-check that this is the case, you can check in the logs that you get the following message:
```
=======================================================
=========== Compiling the Training Function ===========
=======================================================
(Re)Initializing parameters for the optimization. Reason: Uninitialized: [False], Reset requested: [True]
...Initializing attributes for the optimization...
```

The magic phrase here is the indication that a *reset was requested*. Another way to check that optimization was indeed reset is to check in the logs whether the first epoch of the new training session is logged as Epoch #0 (search for the first occurence of the line: `Starting new Epoch!`) 

Note#1: The reset of the optimizer **does not reset the trainable parameters** such as weights and biases. Only reinitializes the optimization, according to the new training config file for session B.

Note#2: If *training session B* is ever interrupted (eg machine dies) and you wish to continue it from an intermediate epoch where it was left, you should NOT use again the `-resetOptimizer` option. Just resume it like normal, as described in Section 3.2 in [main documentation](https://github.com/Kamnitsask/deepmedic/blob/master/documentation/README.md). In the case it was interrupted before even the first epoch is completed, so it essentially needs to be started from scratch, then indeed use the `-resetOptimizer` again.)

### 2. Transfering weights from a previously trained model to a new one

Like in the previous case, assume you have a **modelA** that has been previously trained on **dataA**. You want to learn to segment **dataB**, for which you do have manual labels. *dataA* may differ from *dataB*, for example having different number of MR sequences, or different number of classes. For this, you **need to create a new network**, compatible with dataB. You may even want to create a bigger model. However you wish to utilize the knowledge of previously trained *modelA*, which is relevant for the new task. This can be done by transfering the learnt weights of *modelA* to *modelB* when the latter is created, before we start training it on *dataB*.

This can be performed using the `-pretrained` option in the command line when we create the new model B, and simply pointing to the previously trained and saved *modelA* :
```cshell
python ./deepMedicRun -newModel ./path/to/modelConfigB.cfg \
                      -pretrained ./path/to/previously/trained/model/modelA.trainSessionA.date.save
```

This will create `modelB` according to the configuration file modelConfigB.cfg, just as it is normally done without the `-pretrained` option. Afterwards, however, the parameters of *modelA*, such as weights and biases, are transfered to *modelB*. You should see in the logs indications about the layers that their parameters are being transfered. The modelB with the transfered parameters should now be saved in the output folder (by default at ./examples/output/cnnModels/), with an indication `.pretrained.` in its filename. You can now use this model to train it like normal with a new training session on *dataB*:
```
python ./deepMedicRun -dev gpu -train ./path/to/trainingConfig.cfg -model ./path/to/modelB.pretrained.date.save 
```
Note#1: Make sure that the process exits without errors! Which may occur if the models are incompatible, for instance if number of layers varies. See next subsection about this.

Note#2: Bear in mind that loading/unpickling *modelA* may fail if it was created using an older version of the code than modelB, and thus weights can't be transfered from it. This can happen if the Cnn3d class has been significantly changed between versions. Saving/loading is still not forward-compatible.

Note#3: You do not need to reset the optimizer of *modelB* like in Section 1. *modelB* has not undergone training yet and thus its optimizer is uninitialized anyway.


#### 2.2. Which layers are transfered and how to specify them manually

The above procedure with the `-pretrained` option tries to transfer the kernel weights, biases, and parameters of batch normalization and PreLu is the latter two are used.

By default, the procedure tries to transfer the weights between all the layers except the last one, the classification layer, which is still initialized randomly.

The parameters are transfered according to the following points:
- Each of the parallel multi-scale pathways of *modelB* receives parameters from the corresponding pathway of *modelA*. If *modelB* has more pathways than *modelA*, then the additional pathways of *B* will get the parameters of the *last* pathway of *A*. Eg, if *modelB* has 2 paths and *modelA* just one, then both paths of B will gets params of the 1st path of *modelA*.
- The Fully Connected (FC) pathway (implemented as 1x1x1 convs) of *modelB* get parameters from the FC path of *modelA*.
- If number of kernels in between correpsonding layers of *modelA* and *modelB* differ, the minimum number of kernels is transfered. Eg, if *modelB* is two time wider, half its kernels at each layer will still be randomly initialized.
- It is **required** that the transfered kernels have the same shape (eg 3x3x3) in both models.
- It is **required** that *modelA* has **at least as many layers** as *modelB* in each pathway from which we transfer weights. If not, an error will be returned.
- If the last requirement does not hold, because *modelA* is not deep enough, we can still transfer its weights, but we'll need to specify the transfered layers manually, with care. See below for option `-layers`.

We also provide the option of specifying the layers of **modelA** that should be transferred, by specifying the **layers' depths**. This can be useful if we only wish to transfer certain layers, or the depths of the two nets are not compatible and the default attempt fails. This can be performed by using the option `-layers` along with the `-pretrained` option:
```cshell
python ./deepMedicRun -newModel ./path/to/modelConfigB.cfg \
                      -pretrained ./path/to/previously/trained/model/modelA.trainSessionA.date.save -layers 1 2 5
```

If *modelA* is a 7 layers deep net with two pathways such as [this one](./deepMedic.png), and *modelB* is a [11 layers deep net](./dmRes.png), *assuming* that the kernels of both have the same size (although this does not hold in these figures) the above command would perform the following: The parameters of the two first layers (depths 1, 2) **from each parallel pathway** of *modelA* will be transfered to the corresponding ones in *modelB*. Also, the parameters of the 1st layer of the FC path of *modelA* (depth 5) will be transfered to the 1st layer of the FC path of *modelB*. Note that the given depth 5 refers to the depth of a layer in *modelA*, and is transfered to the corresponding layer in B, which may though be at a different depth.

Note: The code of the transfering procedure can be found at `./deepmedic/cnnTransferParameters.py`.

### 3. Freeze layers of a model when training

In certain conditions it may be desired that certain layers of a networks are kept fixed during training. As an example, we may have a model with parameters trained on *dataA* (can be made with either of the above two methods described in Sec 1 and 2) and now wish to fine-tune with a training session using *dataB* for another task. However, we may only have very few labeled data in database B. In this case, it may be better to freeze some of the layers in the model, for example the freeze the conv layers and train only the layers of the FC path.

We can specify layers to keep frozen during training by specifying them in the configuration file for the training session. For an example see the [trainConfig.cfg of deepmedic](https://github.com/Kamnitsask/deepmedic/blob/master/examples/configFiles/deepMedic/train/trainConfig.cfg):

```
#+++++++Freeze Layers++++++
#[Optional] Specify layers the weights of which you wish to be kept fixed during training (eg to use weights from pre-training). First layer is 1.
# One list for each of the normal, subsampled, and fully-connected (as 1x1 convs) pathways. For instance, provide [1,2,3] to keep first 3 layers fixed. [] or comment entry out to train all layers.
# Defaults: [] for the Normal and FC pathway. For the Subsampled pathway, if entry is not specified, we mirror the option used for the Normal pathway. 
layersToFreezeNormal = []
layersToFreezeSubsampled = []
layersToFreezeFC = []
```

If the above three parameters are not specified or commented out, we do not freeze any layers by default. We can specify to freeze certain layers of a pathway by providing their position within that pathway. For instance:
```
layersToFreezeNormal = [1,2,3]
layersToFreezeFC = [1]
```
The above config will keep the first 3 layers of **both** parallel pathways frozen, as well as the 1st layer in the FC pathway. Notice that if the parameter for the *Subsampled* pathway is not given, such as in the above, **we freeze the same layers as for the normal-resolution pathway**.
