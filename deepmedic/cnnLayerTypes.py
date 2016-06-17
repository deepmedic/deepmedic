# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import numpy
import numpy as np

import theano
import theano.tensor as T

from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d #conv3d2d fixed in bleeding edge version of theano.
import random

from deepmedic.maxPoolingModule import myMaxPooling3d

#is_train is a pseudo boolean (integer) theano variable for switching between training and prediction 
#        self.output = T.switch(T.neq(is_train, 0), train_output, output)

def applyDropout(rng, dropoutRate, inputImageShape, input, inputInferenceBn, inputTestingBn) :
	if dropoutRate > 0.001 : #Below 0.001 I take it as if there is no dropout at all. (To avoid float problems with == 0.0. Although my tries show it actually works fine.)
		probabilityOfStayingActivated = (1-dropoutRate)
		srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
		dropoutMask = srng.binomial(n=1, size=inputImageShape, p=probabilityOfStayingActivated, dtype=theano.config.floatX)
		inputImgAfterDropout = input * dropoutMask
		inputImgAfterDropoutInference = inputInferenceBn * probabilityOfStayingActivated
		inputImgAfterDropoutTesting = inputTestingBn * probabilityOfStayingActivated
	else :
		inputImgAfterDropout = input
		inputImgAfterDropoutInference = inputInferenceBn
		inputImgAfterDropoutTesting = inputTestingBn

	return [inputImgAfterDropout, inputImgAfterDropoutInference, inputImgAfterDropoutTesting]


#Here lets create a new type of convolution layer that will take the place of the old regression-layer class,
#...which should be used as a final layer in a patchwise classifier. It should be a convolution layer + softmax,
#...with a different costfunction, to work with patches.
class ConvLayerWithSoftmax(object):
    """Final Classification layer"""

    def __init__(self,
		rng,
		input,
		inputInferenceBn,
		inputTestingBn,
		image_shape,
		image_shapeValidation,
		image_shapeTesting,
		filter_shape,
		poolsize,
		initializationTechniqueClassic0orDelvingInto1,
		dropoutRate=0.0,
		softmaxTemperature=1):
        """
        type rng: numpy.random.RandomState
        param rng: a random number generator used to initialize weights

        type input:  tensor5 = theano.tensor.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
        param input: symbolic image tensor, of shape image_shape

        type filter_shape: tuple or list of length 5
        param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width, filter depth)

        type image_shape: tuple or list of length 5
        param image_shape: (batch size, num input feature maps,
                             image height, image width, filter depth)

        type poolsize: tuple or list of length 3
        param poolsize: the downsampling (pooling) factor (#rows, #cols, #depth)
        """

	self.inputImageShape = image_shape
	self.inputImageShapeValidation = image_shapeValidation
	self.inputImageShapeTesting = image_shapeTesting
	self.outputImageShape = ""
	self.outputImageShapeValidation = ""
	self.outputImageShapeTesting = ""

	self.activationFunctionType = "" #relu or prelu

        assert image_shape[1] == filter_shape[1]
	self.numberOfFeatureMaps = filter_shape[0]
        self.numberOfOutputClasses = filter_shape[0]

        numberOfPatchesDenselyClassifiedAtOneGo = (image_shape[2]-filter_shape[2]+1)*(image_shape[3]-filter_shape[3]+1)*(image_shape[4]-filter_shape[4]+1)
	numberOfPatchesDenselyClassifiedAtOneGoValidation = (image_shapeValidation[2]-filter_shape[2]+1)*(image_shapeValidation[3]-filter_shape[3]+1)*(image_shapeValidation[4]-filter_shape[4]+1)
	numberOfPatchesDenselyClassifiedAtOneGoTesting = (image_shapeTesting[2]-filter_shape[2]+1)*(image_shapeTesting[3]-filter_shape[3]+1)*(image_shapeTesting[4]-filter_shape[4]+1)

	if initializationTechniqueClassic0orDelvingInto1 == 0 :
		stdForInitialization = 0.01 #this is what I was using for my whole first year.
	elif initializationTechniqueClassic0orDelvingInto1 == 1 :
		stdForInitialization = np.sqrt( 2.0 / (filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4]) ) #Delving Into rectifiers suggestion.

        self.W = theano.shared(
            numpy.asarray(
                rng.normal(loc=0.0, scale=stdForInitialization, size=(filter_shape[0],filter_shape[4],filter_shape[1],filter_shape[2],filter_shape[3])),
                dtype='float32'#theano.config.floatX
            ),
            borrow=True
        )
        filters_shape_conv3d2d = (filter_shape[0], filter_shape[4], filter_shape[1], filter_shape[2], filter_shape[3])

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros ( (filter_shape[0]), dtype = 'float32' )#theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)



	#==============DROPOUT===============
	[self.inputImgAfterDropout,
	self.inputImgAfterDropoutInference,
	self.inputImgAfterDropoutTesting] = applyDropout(rng, dropoutRate, self.inputImageShape, input, inputInferenceBn, inputTestingBn)
	#================End of Dropout section


	#----------CONVOLUTIONS FOR TRAINING---------------
        #Reshape image, image shape, W and filter_shape to be alright for what conv3d2d needs:
        inputImg_reshaped = self.inputImgAfterDropout.dimshuffle(0, 4, 1, 2, 3) #0,4,1,2,3
        image_shape_conv3d2d = (image_shape[0], image_shape[4], image_shape[1], image_shape[2], image_shape[3]) # batch_size, time, num_of_input_channels, rows, columns
        conv_out = T.nnet.conv3d2d.conv3d(signals = inputImg_reshaped, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')
        #Output is in the shape of the input image (signals_shape).
        
	#----------CONVOLUTIONS FOR VALIDATION---------------
	inputImg_reshaped_inference = self.inputImgAfterDropoutInference.dimshuffle(0, 4, 1, 2, 3) #0,4,1,2,3
	image_shape_conv3d2d_validation = (image_shapeValidation[0], image_shapeValidation[4], image_shapeValidation[1], image_shapeValidation[2], image_shapeValidation[3])
        conv_out_inference = T.nnet.conv3d2d.conv3d(signals = inputImg_reshaped_inference, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d_validation, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')

	#----------CONVOLUTIONS FOR TESTING---------------
	inputImg_reshaped_testing = self.inputImgAfterDropoutTesting.dimshuffle(0, 4, 1, 2, 3) #0,4,1,2,3
	image_shape_conv3d2d_testing = (image_shapeTesting[0], image_shapeTesting[4], image_shapeTesting[1], image_shapeTesting[2], image_shapeTesting[3])
        conv_out_testing = T.nnet.conv3d2d.conv3d(signals = inputImg_reshaped_testing, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d_testing, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')


        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x') # Add the biases to the output.
        
        #reshape the result, to have the dimensions as the image_shape that I got in the input. Conv3d2d gives result in another shape, the shape of conv3d2d's input.
        self.output = self.output.dimshuffle(0, 2, 3, 4, 1)#0,2,3,4,1 #result = (batchSize, outputFeatureMapsNumb, x, y, z)

        
	#------------------------SOFTMAX ETC--------------------
        #THIS IS BAD. I NEED TO RESHAPE SO THAT IN FIRST DIMENSION I HAVE ALL PATCHES AND BATCHES, AND THEN RESHAPE BACK?
        #CAUSE SOFTMAX TAKES 2dmatrix. 1dim=batch_size, 2nd = input neurons. Here I have 3 (batchsize,patchesPerImage, inputneurons. Need to reshape)
        outpshuffled = self.output.dimshuffle(0,2,3,4,1)
        outpflattened = outpshuffled.flatten(1)
        first_dim_of_softmax_input = image_shape[0]*numberOfPatchesDenselyClassifiedAtOneGo
        softmax_input = outpflattened.reshape((first_dim_of_softmax_input, self.numberOfOutputClasses))
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        #       this .dot is: [batchSize,prevLayerNeurons] x [prevLayerNeurons, thisLayerNeurons] = [batchSize, thisLayerNeurons]
        #                       thisLayerNeurons = as many as the classes.
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = T.nnet.softmax(softmax_input/softmaxTemperature)
        
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]


	#------------ Same thing, for Validation ("inference" mistakenly named") (different shape, and gets batch normalization from previous layers) -----------

	self.outputInference = conv_out_inference + self.b.dimshuffle('x', 'x', 0, 'x', 'x') # Add the biases to the output.
	self.outputInference = self.outputInference.dimshuffle(0, 2, 3, 4, 1)
	outpshuffled_inference = self.outputInference.dimshuffle(0, 2, 3, 4, 1)
        outpflattened_inference = outpshuffled_inference.flatten(1)
	first_dim_of_softmax_input_validation = image_shapeValidation[0]*numberOfPatchesDenselyClassifiedAtOneGoValidation
	softmax_input_inference = outpflattened_inference.reshape((first_dim_of_softmax_input_validation,self.numberOfOutputClasses))
	self.p_y_given_x_inference = T.nnet.softmax(softmax_input_inference/softmaxTemperature)
	self.y_pred_inference = T.argmax(self.p_y_given_x_inference, axis=1)

	#-------------Different output for TESTING, for using bigger image-part during testing-----------

	self.outputTesting = conv_out_testing + self.b.dimshuffle('x', 'x', 0, 'x', 'x') # Add the biases to the output.
	self.outputTesting = self.outputTesting.dimshuffle(0, 2, 3, 4, 1)
	outpshuffled_testing = self.outputTesting.dimshuffle(0, 2, 3, 4, 1)
        outpflattened_testing = outpshuffled_testing.flatten(1)
	first_dim_of_softmax_input_testing = image_shapeTesting[0]*numberOfPatchesDenselyClassifiedAtOneGoTesting
	softmax_input_testing = outpflattened_testing.reshape((first_dim_of_softmax_input_testing,self.numberOfOutputClasses))
	self.p_y_given_x_testing = T.nnet.softmax(softmax_input_testing/softmaxTemperature)
	self.y_pred_testing = T.argmax(self.p_y_given_x_testing, axis=1)


	#----------save the size of output images for use from other modules---------
	self.outputImageShape = [image_shape[0],
				self.numberOfFeatureMaps,
				image_shape[2]-filter_shape[2]+1,
				image_shape[3]-filter_shape[3]+1,
				image_shape[4]-filter_shape[4]+1]
	self.outputImageShapeValidation = [image_shapeValidation[0],
					self.numberOfFeatureMaps,
					image_shapeValidation[2]-filter_shape[2]+1,
					image_shapeValidation[3]-filter_shape[3]+1,
					image_shapeValidation[4]-filter_shape[4]+1]
	self.outputImageShapeTesting = [image_shapeTesting[0],
					self.numberOfFeatureMaps,
					image_shapeTesting[2]-filter_shape[2]+1,
					image_shapeTesting[3]-filter_shape[3]+1,
					image_shapeTesting[4]-filter_shape[4]+1]

	#NEW CHANGE: Make the reshape to [batchSize,R,C,Z,numberOfFms], instead of doing it in the do_validation_training function outside. CAREFUL. Still different than above, that had FMs as second dimension.
	self.p_y_given_x_testing_reshapedAsImage = self.p_y_given_x_testing.reshape((self.outputImageShapeTesting[0], self.outputImageShapeTesting[2], self.outputImageShapeTesting[3], self.outputImageShapeTesting[4], self.outputImageShapeTesting[1])) #Result: batchSize, R,C,Z, Classes.
	self.p_y_given_x_testing_reshapedAsImage = self.p_y_given_x_testing_reshapedAsImage.dimshuffle(0,4,1,2,3) #Now is (batchSize, FMs, R,C,Z) like input.

    def negativeLogLikelihood(self, y, weightsOfClassesInCostFunction):
        """
        From the theano tutorial: 

        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

#        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
	
	#Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
	e1 = np.finfo(np.float32).tiny
	addTinyProbMatrix = T.lt(self.p_y_given_x, 4*e1) * e1

	weightsOfClassesInCostFunctionDimshuffled = weightsOfClassesInCostFunction.dimshuffle('x',0)
	logOf_p_y_given_x = T.log(self.p_y_given_x + addTinyProbMatrix) #added a tiny so that it does not go to zero and I have problems with nan again...
	weighted_logOf_p_y_given_x = logOf_p_y_given_x * weightsOfClassesInCostFunctionDimshuffled
        return -T.mean( weighted_logOf_p_y_given_x[T.arange(y.shape[0]), y] )

        # end-snippet-2



    def meanErrorTraining(self, y):
        """
        Returns float = number of errors / number of examples of the minibatch ; [0., 1.]

        type y: theano.tensor.TensorType
        param y: corresponds to a vector that gives for each example the
                  correct label
        """
	#Mean error of the training batch.
	tneq = T.neq(self.y_pred, y)
	meanError = T.mean(tneq)
	return meanError

    def meanErrorValidation(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred_inference.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred_inference',
                ('y', y.type, 'y_pred_inference', self.y_pred_inference.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            tneq = T.neq(self.y_pred_inference, y)
            meanError = T.mean(tneq)
            return meanError #The percentage of the predictions that is not the correct class.

        else:
            raise NotImplementedError()

    def realPosAndNegAndTruePredPosNegTraining0OrValidation1(self, y, training0OrValidation1):
	#***Implemented only for binary***. For multiclass, it counts real-positives as everything not-background, and as true positives the true predicted lesion (ind of class).
        vectorOneAtRealPositives = T.gt(y,0)
        vectorOneAtRealNegatives = T.eq(y,0)
	if training0OrValidation1 == 0 : #training:
		yPredToUse = self.y_pred
	else: #validation
		yPredToUse = self.y_pred_inference
        vectorOneAtPredictedPositives = T.gt(yPredToUse,0)
        vectorOneAtPredictedNegatives = T.eq(yPredToUse,0)
        vectorOneAtTruePredictedPositives = T.and_(vectorOneAtRealPositives,vectorOneAtPredictedPositives)
        vectorOneAtTruePredictedNegatives = T.and_(vectorOneAtRealNegatives,vectorOneAtPredictedNegatives)
            
        numberOfRealPositives = T.sum(vectorOneAtRealPositives)
        numberOfRealNegatives = T.sum(vectorOneAtRealNegatives)
        numberOfTruePredictedPositives = T.sum(vectorOneAtTruePredictedPositives)
        numberOfTruePredictedNegatives = T.sum(vectorOneAtTruePredictedNegatives)

	return [ numberOfRealPositives,
                 numberOfRealNegatives,
                 numberOfTruePredictedPositives,
                 numberOfTruePredictedNegatives
               ]

         

    def multiclassRealPosAndNegAndTruePredPosNegTraining0OrValidation1(self, y, training0OrValidation1):
	"""
	The returned list has (numberOfClasses)x4 integers: >numberOfRealPositives, numberOfRealNegatives, numberOfTruePredictedPositives, numberOfTruePredictedNegatives< for each class (incl background).
	Order in the list is the natural order of the classes (ie class-0 RP,RN,TPP,TPN, class-1 RP,RN,TPP,TPN, class-2 RP,RN,TPP,TPN ...)
	"""
	returnedListWithNumberOfRpRnPpPnForEachClass = []

	for class_i in xrange(0, self.numberOfOutputClasses) :
		#Number of Real Positive, Real Negatives, True Predicted Positives and True Predicted Negatives are reported PER CLASS (first for WHOLE).
		vectorOneAtRealPositives = T.eq(y, class_i)
		vectorOneAtRealNegatives = T.neq(y, class_i)

		if training0OrValidation1 == 0 : #training:
			yPredToUse = self.y_pred
		else: #validation
			yPredToUse = self.y_pred_inference

		vectorOneAtPredictedPositives = T.eq(yPredToUse, class_i)
		vectorOneAtPredictedNegatives = T.neq(yPredToUse, class_i)
		vectorOneAtTruePredictedPositives = T.and_(vectorOneAtRealPositives,vectorOneAtPredictedPositives)
		vectorOneAtTruePredictedNegatives = T.and_(vectorOneAtRealNegatives,vectorOneAtPredictedNegatives)
		    
		returnedListWithNumberOfRpRnPpPnForEachClass.append( T.sum(vectorOneAtRealPositives) )
		returnedListWithNumberOfRpRnPpPnForEachClass.append( T.sum(vectorOneAtRealNegatives) )
		returnedListWithNumberOfRpRnPpPnForEachClass.append( T.sum(vectorOneAtTruePredictedPositives) )
		returnedListWithNumberOfRpRnPpPnForEachClass.append( T.sum(vectorOneAtTruePredictedNegatives) )

	return returnedListWithNumberOfRpRnPpPnForEachClass

    def predictionProbabilities(self) :
        return [self.p_y_given_x_testing] 

    def predictionProbabilitiesReshaped1(self) :
        return self.p_y_given_x_testing_reshapedAsImage


    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
        return self.outputTesting[:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]



class ConvLayer(object):

    def __init__(self,
		rng,
		input,
		inputInferenceBn,
		inputTestingBn,
		image_shape,
		image_shapeValidation,
		image_shapeTesting,
		filter_shape,
		poolsize,
		rollingAverageForBatchNormalizationOverThatManyBatches, #If this is 0, it means we are not using BatchNormalization
		maxPoolingParameters,
		initializationTechniqueClassic0orDelvingInto1,
		activationFunctionToUseRelu0orPrelu1=0,
		dropoutRate=0.0):
        """
        type rng: numpy.random.RandomState
        param rng: a random number generator used to initialize weights

        type input:  tensor5 = theano.tensor.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
        param input: symbolic image tensor, of shape image_shape

        type filter_shape: tuple or list of length 5
        param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width, filter depth)

        type image_shape: tuple or list of length 5
        param image_shape: (batch size, num input feature maps,
                             image height, image width, filter depth)

        type poolsize: tuple or list of length 3
        param poolsize: the downsampling (pooling) factor (#rows, #cols, #depth)
        """

	self.inputImageShape = image_shape
	self.inputImageShapeValidation = image_shapeValidation
	self.inputImageShapeTesting = image_shapeTesting
	self.outputImageShape = ""
	self.outputImageShapeValidation = ""
	self.outputImageShapeTesting = ""

	self.activationFunctionType = "" #relu or prelu


        assert image_shape[1] == filter_shape[1]
	self.numberOfFeatureMaps = filter_shape[0]

	if initializationTechniqueClassic0orDelvingInto1 == 0 :
		stdForInitialization = 0.01 #this is what I was using for my whole first year.
	elif initializationTechniqueClassic0orDelvingInto1 == 1 :
		stdForInitialization = np.sqrt( 2.0 / (filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4]) ) #Delving Into rectifiers suggestion.

        self.W = theano.shared(
            numpy.asarray(
                rng.normal(loc=0.0, scale=stdForInitialization, size=(filter_shape[0],filter_shape[4],filter_shape[1],filter_shape[2],filter_shape[3])),
                dtype='float32'#theano.config.floatX
            ),
            borrow=True
        )
        filters_shape_conv3d2d = (filter_shape[0], filter_shape[4], filter_shape[1], filter_shape[2], filter_shape[3])
	"""
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros ( (filter_shape[0]), dtype = 'float32' )
        self.b = theano.shared(value=b_values, borrow=True)
	"""

	#==============DROPOUT===============
	[self.inputImgAfterDropout,
	self.inputImgAfterDropoutInference,
	self.inputImgAfterDropoutTesting] = applyDropout(rng, dropoutRate, self.inputImageShape, input, inputInferenceBn, inputTestingBn)
	#================End of Dropout section


        #Reshape image, image shape, W and filter_shape to be alright for what conv3d2d needs:
        inputImg_reshaped = self.inputImgAfterDropout.dimshuffle(0, 4, 1, 2, 3)
        image_shape_conv3d2d = (image_shape[0], image_shape[4], image_shape[1], image_shape[2], image_shape[3]) # batch_size, time, num_of_input_channels, rows, columns
        conv_out = T.nnet.conv3d2d.conv3d(signals = inputImg_reshaped, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')
        #Output is in the shape of the input image (signals_shape).
        
	#INPUT FOR INFERENCE
	inputImgInference_reshaped = self.inputImgAfterDropoutInference.dimshuffle(0, 4, 1, 2, 3)
        image_shape_conv3d2d_validation = (image_shapeValidation[0], image_shapeValidation[4], image_shapeValidation[1], image_shapeValidation[2], image_shapeValidation[3])
        conv_out_inference = T.nnet.conv3d2d.conv3d(signals = inputImgInference_reshaped, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d_validation, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')
	#INPUT FOR TESTING (has different dimension cause of different image-part to scan image with.
	inputImgTesting_reshaped = self.inputImgAfterDropoutTesting.dimshuffle(0, 4, 1, 2, 3)
        image_shape_conv3d2d_testing = (image_shapeTesting[0], image_shapeTesting[4], image_shapeTesting[1], image_shapeTesting[2], image_shapeTesting[3])
        conv_out_testing = T.nnet.conv3d2d.conv3d(signals = inputImgTesting_reshaped, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = self.W, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = image_shape_conv3d2d_testing, 
                                  filters_shape = filters_shape_conv3d2d,
                                  border_mode = 'valid')


	#-------------------BATCH NORMALIZATION-------------------
	if rollingAverageForBatchNormalizationOverThatManyBatches > 0 :
		gBn_values = np.ones( (filter_shape[0]), dtype = 'float32' )
		self.gBn = theano.shared(value=gBn_values, borrow=True)
		bBn_values = np.zeros( (filter_shape[0]), dtype = 'float32')
		self.bBn = theano.shared(value=bBn_values, borrow=True)

		#for rolling average:
		self.muBnsArrayForRollingAverage = theano.shared(np.zeros( (rollingAverageForBatchNormalizationOverThatManyBatches, filter_shape[0]), dtype = 'float32' )
						,borrow=True)
		self.varBnsArrayForRollingAverage = theano.shared(np.ones( (rollingAverageForBatchNormalizationOverThatManyBatches, filter_shape[0]), dtype = 'float32' )
						,borrow=True)
		self.sharedNewMu_B = theano.shared(np.zeros( (filter_shape[0]), dtype = 'float32'), borrow=True)
		self.sharedNewVar_B = theano.shared(np.ones( (filter_shape[0]), dtype = 'float32'), borrow=True)
		#print "+++++++++++++++++++++++++++++++SHAPE OF THE SHARED VARIABLE self.sharedNewMu_B =======", self.sharedNewMu_B.get_value().shape
		#---

		e1 = np.finfo(np.float32).tiny 
		#WARN, PROBLEM, THEANO BUG. The below was returning (True,) instead of a vector, if I have only 1 FM. (Vector is (False,)). Think I corrected this bug.
		mu_B = conv_out.mean(axis=[0,1,3,4]) #average over all axis but the 2nd, which is the FM axis.
		mu_B = T.unbroadcast(mu_B, (0)) #The above was returning a broadcastable (True,) tensor when FM-number=1. Here I make it a broadcastable (False,), which is the "vector" type. This is the same type with the sharedNewMu_B, which we are updating with this. They need to be of the same type.
		var_B = conv_out.var(axis=[0,1,3,4])
		var_B = T.unbroadcast(var_B, (0))
		var_B_plusE = var_B + e1

		self.newMu_B = mu_B
		self.newVar_B = var_B

		#---computing mu and var for inference from rolling average---
		mu_RollingAverage = self.muBnsArrayForRollingAverage.mean(axis=0)
		effectiveSize = image_shape[0]*image_shape[2]*image_shape[3]*image_shape[4] #batchSize*voxels in a featureMap. See p5 of the paper.
		var_RollingAverage = (effectiveSize/(effectiveSize-1))*self.varBnsArrayForRollingAverage.mean(axis=0)
		var_RollingAverage_plusE = var_RollingAverage + e1

		#OUTPUT FOR TRAINING
		normXi = (conv_out - mu_B.dimshuffle('x', 'x', 0, 'x', 'x')) /  T.sqrt(var_B_plusE.dimshuffle('x', 'x', 0, 'x', 'x')) 
		normYi = self.gBn.dimshuffle('x', 'x', 0, 'x', 'x') * normXi + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x') # dimshuffle makes b broadcastable.
		#OUTPUT FOR INFERENCE
		normXi_inference = (conv_out_inference - mu_RollingAverage.dimshuffle('x', 'x', 0, 'x', 'x')) /  T.sqrt(var_RollingAverage_plusE.dimshuffle('x', 'x', 0, 'x', 'x')) 
		normYi_inference = self.gBn.dimshuffle('x', 'x', 0, 'x', 'x') * normXi_inference + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x')
		#OUTPUT FOR TESTING
		normXi_testing = (conv_out_testing - mu_RollingAverage.dimshuffle('x', 'x', 0, 'x', 'x')) /  T.sqrt(var_RollingAverage_plusE.dimshuffle('x', 'x', 0, 'x', 'x')) 
		normYi_testing = self.gBn.dimshuffle('x', 'x', 0, 'x', 'x') * normXi_testing + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x')

		# Parameters to train for the layer:
		self.params = [self.W, self.gBn, self.bBn] #Careful, if I dont have batch normalization.

	else : #Not using batch normalization
		#make the bias terms. Like the old days.
		bBn_values = np.zeros( (filter_shape[0]), dtype = 'float32')
		self.bBn = theano.shared(value=bBn_values, borrow=True)

		normYi = conv_out + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x')
		normYi_inference = conv_out_inference + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x')
		normYi_testing = conv_out_testing + self.bBn.dimshuffle('x', 'x', 0, 'x', 'x')
		self.params = [self.W, self.bBn]

	#--------------------Activation Function---------------------------------------
	if activationFunctionToUseRelu0orPrelu1 == 0 :
		#print "Layer: Activation function used = ReLu"
		self.activationFunctionType = "relu"
		self.outputBeforeMp = T.maximum(0, normYi)
		self.outputBeforeMpInference = T.maximum(0, normYi_inference)
		self.outputBeforeMpTesting = T.maximum(0, normYi_testing)

	elif activationFunctionToUseRelu0orPrelu1 == 1 :
		#print "Layer: Activation function used = PReLu"
		self.activationFunctionType = "prelu"
		aForPreluValues = np.ones( (filter_shape[0]), dtype = 'float32' )*0.01 #"Delving deep into rectifiers" initializes it like this. LeakyRelus are at 0.01
		self.aForPrelu = theano.shared(value=aForPreluValues, borrow=True) #One separate a (activation) per feature map.
		aForPreluBroadCastedForMultiplWithFms = self.aForPrelu.dimshuffle('x', 'x', 0, 'x', 'x')

		posTraining = T.maximum(0, normYi)
        	negTraining = aForPreluBroadCastedForMultiplWithFms * (normYi - abs(normYi)) * 0.5
		self.outputBeforeMp = posTraining + negTraining
		posInference = T.maximum(0, normYi_inference)
        	negInference = aForPreluBroadCastedForMultiplWithFms * (normYi_inference - abs(normYi_inference)) * 0.5
		self.outputBeforeMpInference = posInference + negInference
		posTesting = T.maximum(0, normYi_testing)
        	negTesting = aForPreluBroadCastedForMultiplWithFms * (normYi_testing - abs(normYi_testing)) * 0.5
		self.outputBeforeMpTesting = posTesting + negTesting

		self.params = self.params + [self.aForPrelu]


	self.outputBeforeMp = self.outputBeforeMp.dimshuffle(0, 2, 3, 4, 1)#reshape the result, to have the dimensions as the image_shape that I got in the input.
	self.outputBeforeMpInference = self.outputBeforeMpInference.dimshuffle(0, 2, 3, 4, 1)
	self.outputBeforeMpTesting = self.outputBeforeMpTesting.dimshuffle(0, 2, 3, 4, 1)

	#----------Save the size of output images for use from other modules---------
	self.outputImageShapeBeforeMp = [image_shape[0],
				filter_shape[0],
				image_shape[2]-filter_shape[2]+1,
				image_shape[3]-filter_shape[3]+1,
				image_shape[4]-filter_shape[4]+1]
	self.outputImageShapeBeforeMpValidation = [image_shapeValidation[0],
					filter_shape[0],
					image_shapeValidation[2]-filter_shape[2]+1,
					image_shapeValidation[3]-filter_shape[3]+1,
					image_shapeValidation[4]-filter_shape[4]+1]
	self.outputImageShapeBeforeMpTesting = [image_shapeTesting[0],
					filter_shape[0],
					image_shapeTesting[2]-filter_shape[2]+1,
					image_shapeTesting[3]-filter_shape[3]+1,
					image_shapeTesting[4]-filter_shape[4]+1]


	#==========MAX POOLING================
	self.maxPoolingParameters = maxPoolingParameters
	if maxPoolingParameters == [] : #no max pooling after this layer.
		self.outputImageShape = self.outputImageShapeBeforeMp
		self.outputImageShapeValidation = self.outputImageShapeBeforeMpValidation
		self.outputImageShapeTesting = self.outputImageShapeBeforeMpTesting

		self.output = self.outputBeforeMp
		self.outputInference = self.outputBeforeMpInference
		self.outputTesting = self.outputBeforeMpTesting
	else : #Max pooling is actually happening here...
		(self.output, self.outputImageShape) = myMaxPooling3d(self.outputBeforeMp, self.outputImageShapeBeforeMp, self.maxPoolingParameters)
		(self.outputInference, self.outputImageShapeValidation) = myMaxPooling3d(self.outputBeforeMpInference, self.outputImageShapeBeforeMpValidation, self.maxPoolingParameters)
		(self.outputTesting, self.outputImageShapeTesting) = myMaxPooling3d(self.outputBeforeMpTesting, self.outputImageShapeBeforeMpTesting, self.maxPoolingParameters)
		print "--------MAX POOLING-------- was done for layer THIS layer. New dimensions of the output after it:"
		print "--------MAX POOLING-------- self.outputImageShape = ", self.outputImageShape
		print "--------MAX POOLING-------- self.outputImageShapeValidation = ", self.outputImageShapeValidation
		print "--------MAX POOLING-------- self.outputImageShapeTesting = ", self.outputImageShapeTesting

    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
	return self.outputTesting[:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]


