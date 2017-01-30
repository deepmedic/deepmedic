# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import theano
import theano.tensor as T
from theano.tensor.signal import pool

from math import ceil

def mirrorFinalBordersOfImage(image3dBC012, mirrorFinalBordersForThatMuch) :
    image3dBC012WithMirroredFinalElemets = image3dBC012
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[0]) :
        image3dBC012WithMirroredFinalElemets = T.concatenate([ image3dBC012WithMirroredFinalElemets, image3dBC012WithMirroredFinalElemets[:,:,-1:,:,:] ], axis=2)
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[1]) :
        image3dBC012WithMirroredFinalElemets = T.concatenate([ image3dBC012WithMirroredFinalElemets, image3dBC012WithMirroredFinalElemets[:,:,:,-1:,:] ], axis=3)
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[2]) :
        image3dBC012WithMirroredFinalElemets = T.concatenate([ image3dBC012WithMirroredFinalElemets, image3dBC012WithMirroredFinalElemets[:,:,:,:,-1:] ], axis=4)
    return image3dBC012WithMirroredFinalElemets


def myMaxPooling3d(image3dBC012, image3dBC012Shape, maxPoolingParameters) :
    # image3dBC012 dimensions: (batch, fms, r, c, z)
    # maxPoolingParameters: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    
    ds = maxPoolingParameters[0]
    stride = maxPoolingParameters[1]
    mode1 = maxPoolingParameters[3]
    
    image3dBC012WithMirroredFinalElemets = mirrorFinalBordersOfImage(image3dBC012, maxPoolingParameters[2])
    
    pooled_out1 = pool.max_pool_2d(
                            input = image3dBC012WithMirroredFinalElemets,
                            ds=(ds[1], ds[2]),
                            ignore_border=True,
                            st=(stride[1],stride[2]),
                            padding=(0, 0),
                            mode=mode1)
    rLastPooledOut1 = pooled_out1.dimshuffle(0,1,3,4,2)
    
    pooled_out2 = pool.max_pool_2d(
                            input = rLastPooledOut1,
                            ds=(1,ds[0]),
                            ignore_border=True,
                            st=(1,stride[0]),
                            padding=(0, 0),
                            mode=mode1)
    pooled_out = pooled_out2.dimshuffle(0,1,4,2,3)
    
    #calculate the shape of the image after the max pooling.
    #This calculation is for ignore_border=True! Pooling should only be done in full areas in the mirror-padded image.
    shapeOfImageAfterMaxPoolingAfterMirroring = [   image3dBC012Shape[0],
                                                    image3dBC012Shape[1],
                                                    int(ceil( (image3dBC012Shape[2] + maxPoolingParameters[2][0] - ds[0] + 1) / (1.0*stride[0])) ),
                                                    int(ceil( (image3dBC012Shape[3] + maxPoolingParameters[2][1] - ds[1] + 1) / (1.0*stride[1])) ),
                                                    int(ceil( (image3dBC012Shape[4] + maxPoolingParameters[2][2] - ds[2] + 1) / (1.0*stride[2])) )
                                                ]
    return (pooled_out, shapeOfImageAfterMaxPoolingAfterMirroring)

