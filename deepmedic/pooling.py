# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
from math import ceil
import theano.tensor as T

def mirrorFinalBordersOfImage(image3dBC012, mirrorFinalBordersForThatMuch) :
    image3dBC012WithMirrorPad = image3dBC012
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[0]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,-1:,:,:] ], axis=2)
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[1]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,-1:,:] ], axis=3)
    for time_i in xrange(0, mirrorFinalBordersForThatMuch[2]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,:,-1:] ], axis=4)
    return image3dBC012WithMirrorPad


def pooling3d(image3dBC012, image3dBC012Shape, poolParams) :
    # image3dBC012 dimensions: (batch, fms, r, c, z)
    # poolParams: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    ws = poolParams[0] # window size
    stride = poolParams[1] # stride
    mode1 = poolParams[3] # max, sum, average_inc_pad, average_exc_pad
    
    image3dBC012WithMirrorPad = mirrorFinalBordersOfImage(image3dBC012, poolParams[2])
    
    T.signal.pool.pool_3dd( input=image3dBC012WithMirrorPad,
                            ws=ws,
                            ignore_border=True,
                            st=stride,
                            pad=(0,0,0),
                            mode=mode1)
    
    #calculate the shape of the image after the max pooling.
    #This calculation is for ignore_border=True! Pooling should only be done in full areas in the mirror-padded image.
    imgShapeAfterPoolAndPad = [ image3dBC012Shape[0],
                                image3dBC012Shape[1],
                                int(ceil( (image3dBC012Shape[2] + poolParams[2][0] - ds[0] + 1) / (1.0*stride[0])) ),
                                int(ceil( (image3dBC012Shape[3] + poolParams[2][1] - ds[1] + 1) / (1.0*stride[1])) ),
                                int(ceil( (image3dBC012Shape[4] + poolParams[2][2] - ds[2] + 1) / (1.0*stride[2])) )
                            ]
    return (pooled_out, imgShapeAfterPoolAndPad)

