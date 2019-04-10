# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:45:37 2018

@author: pi96doc
"""
from pkg_resources import resource_filename
from . import util
from . import config

global JNIUS_RUNNING
if ~('JNIUS_RUNNING' in globals()):
    JNIUS_RUNNING=0

if (JNIUS_RUNNING==0):
    fn=resource_filename("NanoImagingPack","resources/View5D_.jar")
    try:
        import jnius_config
        jnius_config.add_classpath(fn)
        JNIUS_RUNNING = 1
#        jnius_config.add_classpath('C:/Users/pi96doc/Documents/Programming/PythonScripts/view5d.jar')
    except:
        print("Problem setting classpath. Switching to conventional viewer by setting __DEFAULTS__['IMG_VIEWER'] = 'NIP_VIEW' ")
        config.setDefault('IMG_VIEWER', 'NIP_VIEW')

# jnius_config.add_options('-Xrs', '-Xmx4096')
# jnius_config.set_classpath()
# , 'C:/Users/pi96doc/Documents/Programming/PythonScripts/*'
import numpy as np
#import view 

lastviewer=None
global allviewers
allviewers=[]

def v5ProcessKeys(out,KeyList):
#    import time
    for k in KeyList:
        out.ProcessKeyMainWindow(k)
#        time.sleep(0.1)
        out.UpdatePanels()
        out.repaint()

def v5(data,SX=1200,SY=1200,multicol=None,gammaC=0.15,showPhases=False):
    '''
        lauches a java-based viewer view5d
        Since the viewer is based on calling java via the pyjnius Java bridge, it runs in its own thread each time. This causes overhead and may be not advisable for large
        datasets, but it is nice for debugging purposes as it can also display the data within the pdg debugger.
        -----
        data : multidimensional data to display. This can be up to 5d data
        SX,SY : size of the total viewer in pixels to use (default: 1200,1200)
        mulitcolor : If not None or False, the viewer will be launched in a multicolor (RGB) mode by default. Default: None
        gammaC   :  gamma to use for the display of the complex valued magnitude (default: 0.15)
        showPhases : if True, the phases are shown in multiplicative color overlay with a cyclic colormap
          
        Example:
        import NanoImagingPack as nip
        v=nip.v5(np.random.rand(10,10,10,4),multicol=True)
    '''
#    data=np.transpose(data) # force a cast to np.array
    import jnius as jn

    print("Lauching View5D: with datatype: "+str(data.dtype))
    VD = jn.autoclass('view5d.View5D')
#    data=expanddim(data,5) # casts all arrays to 5D
    sz=data.shape
    sz=sz[::-1] # reverse
    sz=np.append(sz,np.ones(5-len(data.shape)))
    #    data=np.transpose(data) # reverts all axes to common display direction
#    data=np.moveaxis(data,(4,3,2,1,0),(0,1,2,3,4)) # reverts axes to common display direction
    if not isinstance(data,np.ndarray):
        try:
            import tensorflow as tf
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                data=data.eval()
            if not isinstance(data,np.ndarray):
                raise ValueError("v5: cannot evaluate tensor.")
        except ImportError:
            raise ValueError("v5: unsupported datatype to display")

    if (data.dtype=='complex' or data.dtype=='complex128' or data.dtype=='complex64'):
        dcr=data.real.flatten()
        dci=data.imag.flatten()
        #dc=np.concatenate((dcr,dci)).tolist();
        dc=np.stack((dcr,dci),axis=1).flatten().tolist()
        #        dc.reverse()
        # dc=data.flatten().tolist();
        out = VD.Start5DViewerC(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)
        #        out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");
        for E in range(int(sz[3])):
            out.SetGamma(E,gammaC)
        out.UpdatePanels()
        if showPhases:
            for E in range(int(sz[3])):
                if sz[3]>1:
                    anElem = util.subslice(data, -4, E)
                else:
                    anElem=data
                phases = (np.angle(anElem)+np.pi)/np.pi*128
                out.AddElement(phases.flatten().tolist(),sz[0],sz[1],sz[2],sz[3],sz[4])   # add phase information to data
                v5ProcessKeys(out,'E') # advance to next element to the just added phase-only channel
                v5ProcessKeys(out,12*'c') # toggle color mode 12x to reach the cyclic colormap
                v5ProcessKeys(out,'vVe') # Toggle from additive into multiplicative display
    else:
        dc=data.flatten().tolist()
        out=None
        if data.dtype == 'float':
            out = VD.Start5DViewerF(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'float32':
            out = VD.Start5DViewerF(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'float64':
            out = VD.Start5DViewerD(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'int':
            out = VD.Start5DViewerI(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint8':
            out = VD.Start5DViewerB(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'bool':
            dc=data.flatten().astype('uint8').tolist()
            out = VD.Start5DViewerB(dc, sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
            out.ProcessKeyMainWindow("r")
        elif data.dtype == 'int16':
            out = VD.Start5DViewerS(dc, sz[0], sz[1], sz[2], sz[3], sz[4], SX, SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint16':
            dc = data.astype("int").flatten().tolist()
            out = VD.Start5DViewerI(dc, sz[0], sz[1], sz[2], sz[3], sz[4], SX, SY)  # calls the WRONG entry point to the Java program
#            out = VD.Start5DViewerUS(dc, sz[0], sz[1], sz[2], sz[3], sz[4], SX, SY)  # calls the WRONG entry point to the Java program
        elif data.dtype == 'uint32' or data.dtype == 'int32':
            out = VD.Start5DViewerI(dc, sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)  # calls the WRONG entry point to the Java program
        else:
            print("View5D: unknown datatype: "+str(data.dtype))
            return None
        v5ProcessKeys(out,'v') # remove first channel from overlay
    if (multicol is None or multicol is False) and (sz[3] > 2 and sz[3] < 5):  # reset the view to single colors
        v5ProcessKeys(out,'CGeGveGvEEv') # set first 3 colors to gray scale and remove from color overlay 
    if (multicol is None or multicol is False) and (sz[3] == 2):  # reset the view to single colors
        v5ProcessKeys(out,'CGeGvEv')

    if not data.name is None:
        out.NameWindow(data.name)
    else:
        name = util.caller_args()[0]
        out.NameWindow(name)

    if not data.pixelsize is None:
        pxs = util.expanddimvec(data.pixelsize,5,trailing=True)
        Names = ['X','Y','Z','E','T']
        if (not data.unit is None) and (type(data.unit) == list) and len(data.unit)>4:
            Units = data.unit[0:5]
        else:
            Units = [data.unit,data.unit,data.unit,'ns','s']
        SV = 1.0  # value scale
        NameV = 'intensity'
        UnitV = 'photons'
        out.SetAxisScalesAndUnits(0, SV, pxs[0], pxs[1], pxs[2], pxs[3], pxs[4], 0, 0, 0, 0, 0, 0, NameV, Names, UnitV, Units)

    if not data.dim_description is None:
        names = None
        if type(data.dim_description) == str:
            names = [data.dim_description]
        else:
            names = data.dim_description
        e=0
        for name in names:
            if e < sz[3]: # has to be a legal element
                out.NameElement(e, name)
                e += 1
    # else:   # no name was given. Try to recover the caller's name

    v5ProcessKeys(out, '12') # to trigger the display update
    out.UpdatePanels()
    out.repaint()
    jn.detach()
    global allviewers
    allviewers.append(out)
    return out

def v5NameElement(out, enum, Name):
    out.NameElement(enum.Name)

def v5close(aviewer=None):
    global allviewers
    if aviewer==None:
        for v in allviewers:
#            print("closing")
            v.closeAll()
        for n in range(len(allviewers)):
            allviewers.pop()
    else:
        import numbers
        if isinstance(aviewer, numbers.Number):
            aviewer=allviewers[aviewer]
            aviewer.closeAll()
            allviewers.remove(aviewer)
        else:
            aviewer.closeAll()
            allviewers.remove(aviewer)

def getMarkers(myviewer=None,ListNo=0, OnlyPos=True):
    markers=[]
    if myviewer == None:
        myviewer = lastviewer
        
    if myviewer != None:
        markers=myviewer.ExportMarkers(ListNo)
        markers=np.array(markers)
        if OnlyPos:
            if len(markers)>0:
                markers=markers[:,4::-1] # extract only position information and orient if for numpy standard
    return markers