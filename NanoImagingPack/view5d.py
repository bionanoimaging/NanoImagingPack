# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:45:37 2018

@author: pi96doc
"""
from pkg_resources import resource_filename
from NanoImagingPack import expanddim

global JNIUS_RUNNING
if ~('JNIUS_RUNNING' in globals()):
    JNIUS_RUNNING=0;

if (JNIUS_RUNNING==0):
    fn=resource_filename("NanoImagingPack","resources/View5D_.jar")
    import jnius_config
    try:
        jnius_config.add_classpath(fn)
#        jnius_config.add_classpath('C:/Users/pi96doc/Documents/Programming/PythonScripts/view5d.jar')
    except:
        print("Problem setting classpath. Continuing...")    
    JNIUS_RUNNING=1;
    
# jnius_config.add_options('-Xrs', '-Xmx4096')
# jnius_config.set_classpath()
# , 'C:/Users/pi96doc/Documents/Programming/PythonScripts/*'
import numpy as np
#import view 

lastviewer=None

def v5(data,SX=1200,SY=1200):
#    data=np.transpose(data) # force a cast to np.array
    import jnius as jn

    print("Lauching View5D: with datatype: "+str(data.dtype))
    VD = jn.autoclass('view5d.View5D')
    data=expanddim(data,5) # casts all arrays to 5D
    sz=data.shape
#    sz=sz[::-1] # reverse
#    sz=np.append(sz,np.ones(5-len(data.shape)));
    data=np.transpose(data) # reverts all axes to common display direction
#    data=np.moveaxis(data,(4,3,2,1,0),(0,1,2,3,4)) # reverts axes to common display direction
    if (data.dtype=='complex'):
        dcr=data.real.flatten();
        dci=data.imag.flatten();
        #dc=np.concatenate((dcr,dci)).tolist();
        dc=np.stack((dcr,dci),axis=1).flatten().tolist();
#        dc.reverse()
        # dc=data.flatten().tolist();
        out = VD.Start5DViewerC(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY);
        out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");
        out.SetGamma(0,0.3)
        out.UpdatePanels()
    else:
        dc=data.flatten().tolist();
        out=None
        if data.dtype == 'float':
            out = VD.Start5DViewerF(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'float32':
            out = VD.Start5DViewerF(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'float64':
            out = VD.Start5DViewerD(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'int':
            out = VD.Start5DViewerI(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint8':
            out = VD.Start5DViewerB(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint16' or data.dtype == 'int16':
            out = VD.Start5DViewerS(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint32' or data.dtype == 'int32':
            out = VD.Start5DViewerI(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY); # calls the WRONG entry point to the Java program
        else:
            print("View5D: unknown datatype: "+str(data.dtype))
            return Null
    out.ProcessKeyMainWindow("1");
    out.ProcessKeyMainWindow("2");
    out.UpdatePanels()
    out.repaint()
    jn.detach();
    lastviewer=out
    return out

def getMarkers(myviewer=None,ListNo=0, OnlyPos=True):
    markers=[]
    if myviewer == None:
        myviewer = lastviewer
        
    if myviewer != None:
        markers=myviewer.ExportMarkers(ListNo)
        markers=np.array(markers)
        if OnlyPos:
            markers=markers[:,0:5] # extract only position information
    return markers