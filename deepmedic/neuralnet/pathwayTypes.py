# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

#################################################################
#                        Types of Pathways                      #
#################################################################

# Also see module deepmedic.neuralnet.pathways.

from __future__ import absolute_import, print_function, division

class PathwayTypes(object):
    NORM = 0; SUBS = 1; FC = 2 # static
    
    def pTypes(self): #To iterate over if needed.
        # This enumeration is also the index in various datastructures ala: [ [listForNorm], [listForSubs], [listForFc] ] 
        return [self.NORM, self.SUBS, self.FC]