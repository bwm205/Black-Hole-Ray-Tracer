#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds all user interaction functions for use in project

CONTAINS
    Function : select_file
"""

import sys

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def select_file(filetype):
    '''Facilitates file selection with GUI. Exits code if no file selected
    
    INPUT
        filetype : String giving the type of file to select - search restricted accordingly
            Choose from 'image' or 'CSV'
        
    OUTPUT
        filename : String giving path to file'''
        
    filetype_options = {'image':[('JPEG ', '*.jpg'),('TIFF','*.tif'), ('Portable Network Graphics','*.png'),('Windows Bitmaps','*.bmp')],
                        'CSV':[('CSV ', '*.csv')]}
    
    Tk().withdraw() # keep the root window from appearing
    filename  = askopenfilename(filetypes = filetype_options[filetype]) # show an "Open" dialog box and return the path to the selected file
    
    #If no file given, exit system
    if len(filename) == 0:
        print('No file selected. Exiting system.')
        sys.exit()
        
    return filename