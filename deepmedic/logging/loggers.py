# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import datetime


def get_pattern_string(pattern='#', width=80):
    if width == 0:
        return ''

    string = pattern * (width // len(pattern))
    leftover = width % len(pattern)

    if leftover != 0:
        string += pattern[:leftover]

    return string


class Logger:
    loggerFileName = None
    
    def print3(self, string):
        print(string)    
        f = open(self.loggerFileName, 'a')
        now = datetime.datetime.now()
        now_str = "{0}-{1}-{2} {3}:{4}:{5:.2f}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                       (now.second + now.microsecond/10**6))
        f.write(now_str + ": " + string + "\n")
        f.close()

    def print_pattern_line(self, pattern='#', line_width=80):
        self.print3(get_pattern_string(pattern=pattern, width=line_width))

    def print_block(self, text, pattern='#', line_width=80, block=True, margin=5):
        """
        Prints a block of text with a character pattern surrounding it.
        :str text: the text to be written
        :str pattern: the character pattern to surround the text with (default '#')
        :int line_width: the total width of the line to be written (default 80)
        :bool block: whether to add a line of pattern only below and above the text block (default True)
        :int margin: the length of the margin (default 5)
        """

        if block:
            self.print_pattern_line(pattern, line_width)

        text_width = line_width - margin * 2 - 2

        if len(text) > text_width:
            long_block = True
        else:
            long_block = False

        # get paragraphs
        paragraphs = text.split('\n')

        if long_block or len(paragraphs) > 1:  # left aligned
            for paragraph in paragraphs:
                # get words
                text_split = paragraph.split()
                # while there is still text to write
                while len(text_split) > 0:
                    # left margin
                    string = get_pattern_string(pattern=pattern, width=margin)
                    text_len = 0
                    i = 0
                    while i < len(text_split) and text_len + len(text_split[i]) + i < text_width:
                        text_len += len(text_split[i])
                        i += 1

                    text_len += max(i - 1, 0)

                    if i == 0:  # word is longer than the available text width. Must split
                        text_split.insert(1, text_split[0][text_width-1:])
                        text_split[0] = text_split[0][:text_width-1] + '-'
                        text_len = text_width
                        i = 1

                    # add text
                    string += ' ' + ' '.join(text_split[:i])
                    # add whitespace
                    string += ' ' * (text_width - text_len + 1)
                    # add right margin
                    string += get_pattern_string(pattern, width=line_width)[-margin:]

                    self.print3(string)

                    if i >= len(text_split):
                        break
                    else:
                        text_split = text_split[i:]
        else:  # centre align
            left_space = (text_width - len(text)) // 2
            right_space = text_width - left_space - len(text)
            string = get_pattern_string(pattern=pattern, width=margin+left_space)
            string += ' ' + text + ' '
            string += get_pattern_string(pattern=pattern, width=line_width)[-(margin+right_space):]

            self.print3(string)

        if block:
            self.print_pattern_line(pattern, line_width)
        
    def __init__(self, log_path="logs/defaultLogFile.txt"):
        self.loggerFileName = log_path
        self.print3("=============================== logger created =======================================")
