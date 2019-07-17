# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2019 Dr. Helder Marchetto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import socket
import time
import numpy as np
import struct
import sys
import readUview as ru
import matplotlib.pyplot as plt
from skimage import exposure


class imgStackClass():
    def __init__(self,fn) -> None:
        """Initializes the imgStackClass object"""
        #fn must be a list of file names to load as a numpy array
        if len(fn) < 1:
            print("Failed. No file loaded")
            return
        try:
            self.imageWidth  = 0
            self.imageHeight = 0
            self.nImages = len(fn)
            for f in fn:
                ind = fn.index(f)
                if ind == 0:
                    ruObj = ru.readUviewClass()
                    img = ruObj.getImage(f)
                    self.imgType = img.dtype.name
                    self.imageWidth  = ruObj.imageWidth
                    self.imageHeight = ruObj.imageHeight
                    self.stack = np.zeros((self.nImages,self.imageWidth,self.imageHeight),dtype=np.ndarray)
                    self.stack[0] = img
                    self.limits = np.percentile(img,(2,98))
#                    print("Loading image nr=%04i" %ind, end='\r')
                    print(f"\rLoading image nr="+str("%04i" %ind),end="")
                    sys.stdout.flush()
                else:
                    img = ruObj.getImage(f)
                    self.imageWidth  = ruObj.imageWidth
                    self.imageHeight = ruObj.imageHeight
                    try:
                        self.stack[ind] = img
#                        print("Loading image nr=%04i" %ind, end='\r')
                        print(f"\rLoading image nr="+str("%04i" %ind),end="")
                        sys.stdout.flush()
                    except:
                        raise Exception("Error loading image nr".format(fn.index(f)))
            print("\n")
            self.current = 0
            self.fn = fn
            self.rawfn = []
            for f in fn:
                self.rawfn.append(os.path.basename(os.path.abspath(f))+'.dat')
            self.dir = os.path.dirname(os.path.abspath(self.fn[0]))
        except:
            print("Loading of images failed")
            return

    def getImage(self,pos=-1) -> np.ndarray:
        if pos < 0:
            pos = self.current
        try:
            return self.stack[pos].astype(self.imgType)
        except:
            raise Exception('Index not valid or stack not yet defined')

    def getLimits(self,pos=-1,clip=2) -> tuple:
        if pos < 0:
            pos = self.current
        try:
            self.limits = np.percentile(self.getImage(pos),(clip,100-clip))
            return self.limits
        except:
            raise Exception('Index not valid or stack not yet defined')

    def getDrawImage(self,pos=-1,clip=2) -> np.ndarray:
        if pos < 0:
            pos = self.current
        if clip != 2:
            limits = np.percentile(self.getImage(pos),(2,98))
        else:
            limits = self.limits
        try:
            img = exposure.rescale_intensity(self.stack[pos].astype(self.imgType), in_range=(limits[0], limits[1]))
            return img
        #.astype(self.imgType)
        except:
            raise Exception('Index not valid or stack not yet defined')

    def __repr__(self):
        try:
            outStr = [str("nImages = %04i" %self.nImages)]
            outStr.append(str("First image = %s" %self.rawfn[0]))
            outStr.append(str("Last image = %s" %self.rawfn[-1]))
            outStr.append(str("Directory = %s" %self.dir))
            outStr.append(str("Image size = (%i,%i)" %(self.imageWidth, self.imageHeight)))
            return '\n'.join(outStr)
        except AttributeError:
            return "Object not defined"

class elmitecAnalysisClass():
    def __init__(self) -> None:
        """Initializes the elmitecAnalysisClass object"""

    def __repr__(self):
        try:
            outStr = [str("nImages = %04i" %self.imgStack.nImages)]
            outStr.append(str("First image = %s" %self.imgStack.fn[0]))
            outStr.append(str("Last image = %s" %self.imgStack.fn[-1]))
            outStr.append(str("Directory = %s" %self.imgStack.dir))
            outStr.append(str("Image size = (%i,%i)" %(self.imageWidth, self.imageHeight)))
            return '\n'.join([outStr])
        except AttributeError:
            return "Object not defined"

    def loadDir(self,dirName):
        self.dir = dirName

import os
def getDatFilesInDir(mypath = r'K:\Data\TurningPointResolution'):
    fileList = []
    for file in os.listdir(mypath):
        if file.endswith(".dat"):
            fileList.append(os.path.join(mypath, file))
    return fileList
fn = getDatFilesInDir()
stack = imgStackClass(fn)
limits= stack.getLimits(clip=2)
plt.imshow(stack.getImage(0), cmap=plt.cm.gray, vmin=limits[0], vmax=limits[1])