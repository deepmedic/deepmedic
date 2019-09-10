# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os


class ConfigData(object):
    def __init__(self):
        self.sections = {}
        self.curr_section = None
        self.num_sections = 0

    def __getitem__(self, item):
        return self.sections[item] if item in self.sections else None

    def get_elem_section(self, name):
        elem = self.get_elem(name)
        if elem:
            return elem.section
        else:
            return None

    def get_elem(self, name):
        for _, section in self.sections.items():
            for elem in section.elems:
                if elem[0].name == name:
                    return elem[0]
        return None

    def set_curr_section(self, section_name, text=None, idx=None, advanced=False):
        if not self.is_section(section_name):
            if not idx:
                idx = self.num_sections
            self.add_section((section_name, idx, text, advanced))
            if not section_name == 'other':
                self.num_sections += 1
        self.curr_section = self.sections[section_name]

    def is_section(self, section_name):
        return section_name in self.sections.keys()

    def add_section(self, section_args):
        self.sections[section_args[0]] = ConfigSection(*section_args)

    def get_sorted_sections(self):
        return sorted(self.sections.values(), key=lambda x: x.get_idx())

    def get_children(self, parent_name):
        children = []
        for _, section in self.sections.items():
            for elem in section.elems:
                if elem[0].parent and elem[0].parent.name == parent_name:
                    children += [elem[0]]
        return children

    def get_elems_with_children(self):
        elems_with_children = {}
        for _, section in self.sections.items():
            for elem in section.elems:
                if elem[0].parent:
                    if elem[0].parent in elems_with_children.keys():
                        elems_with_children[elem[0].parent].append(elem[0])
                    else:
                        elems_with_children[elem[0].parent] = [elem[0]]
        return elems_with_children

    def add_elem(self, name, elem_type='Numeric', widget_type='lineedit',
                 description=None, required=False, options=None, default=None, info=None, advanced=False,
                 parent=None):
        if self.curr_section:
            return self.curr_section.add_elem(ConfigElem(name, idx=self.curr_section.num_elems,
                                                         section=self.curr_section,
                                                         elem_type=elem_type, widget_type=widget_type,
                                                         description=description,
                                                         required=required, options=options, default=default,
                                                         info=info, advanced=advanced, parent=parent))
        else:
            raise Exception('No Section Selected')


class ConfigSection(object):
    def __init__(self, name, idx, text=None, advanced=False):
        self.name = name
        if not text:
            text = name
        self.text = text
        self.idx = idx
        self.elems = []
        self.num_elems = 0
        self.advanced = advanced

    def get_idx(self):
        return self.idx

    def add_elem(self, elem):
        self.elems += [(elem, elem.idx)]
        return elem

    def get_sorted_elems(self):
        return [elem[0] for elem in sorted(self.elems, key=lambda tup: tup[1])]


class ConfigElem(object):
    def __init__(self, name, elem_type='Numeric', widget_type='lineedit', description=None, required=False,
                 options=None, default=None, idx=None, section=None, info=None, advanced=False, parent=None):
        if elem_type == 'Bool':
            widget_type = 'checkbox'
        elif widget_type == 'checkbox':
            elem_type = 'Bool'
        self.name = name
        self.elem_type = elem_type
        if description:
            self.description = description
        else:
            self.description = name
        self.required = required
        self.options = options
        self.default = default
        self.section = section
        self.idx = idx
        self.widget_type = widget_type
        self.info = info
        self.advanced = advanced
        self.parent = parent

    def __get__(self, instance, owner):
        return self.name


class Config(object):
    
    def __init__(self, abs_path_to_cfg):
        self._configStruct = {}
        self._abs_path_to_cfg = abs_path_to_cfg # for printing later.
        print("Given configuration file: ", self._abs_path_to_cfg)
        exec(open(self._abs_path_to_cfg).read(), self._configStruct)
        self._check_for_deprecated_cfg()
        
    def __getitem__(self, key):  # overriding the [] operator.
        return self.get(key)
    
    def get(self, string1):
        return self._configStruct[string1] if string1 in self._configStruct else None
    
    def get_abs_path_to_cfg(self):
        return self._abs_path_to_cfg
    
    def _check_for_deprecated_cfg(self):
        pass
    
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        pass
    

