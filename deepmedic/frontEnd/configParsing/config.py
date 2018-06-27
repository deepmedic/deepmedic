# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

class Config(object):
    
    def __init__(self, abs_path_to_cfg):
        self._configStruct = {}
        self._abs_path_to_cfg = abs_path_to_cfg # for printing later.
        print("Given configuration file: ", self._abs_path_to_cfg)
        exec(open(self._abs_path_to_cfg).read(), self._configStruct)
        self._check_for_deprecated_cfg()
        
    def __getitem__(self, key): # overriding the [] operator.
        return self.get(key)
    
    def get(self, string1) :
        return self._configStruct[string1] if string1 in self._configStruct else None
    
    def get_abs_path_to_cfg(self):
        return self._abs_path_to_cfg
    
    def _check_for_deprecated_cfg(self):
        pass
    
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        pass
    

