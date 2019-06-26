# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:45:37 2018

@author: pi96doc
"""
from pkg_resources import resource_filename
from . import util
from . import transformations
from . import config
from . import image

global JNIUS_RUNNING
if ~('JNIUS_RUNNING' in globals()):
    JNIUS_RUNNING=0

if (JNIUS_RUNNING==0):
    fn=resource_filename("NanoImagingPack","resources/View5D_.jar")
    try:
        import jnius_config
#        jnius_config.add_options('-Xrs', '-Xmx4096')
        jnius_config.add_options('-Dswing.aatext=true', '-Dswing.plaf.metal.controlFont=Tahoma', '-Dswing.plaf.metal.userFont=Tahoma', '-Dawt.useSystemAAFontSettings=on')
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
def vv(data, SX=1200, SY=1200, multicol=None, gamma=None, showPhases=False, fontSize=18, linkElements = None):
    if config.__DEFAULTS__['IMG_VIEWER'] == 'VIEW5D':
        return v5(data, SX=SX, SY=SY, multicol=multicol, gamma=gamma, showPhases=showPhases, fontSize=fontSize, linkElements = linkElements)
    else:
        return data._repr_pretty_([], cycle=False)

def v5(data, SX=1200, SY=1200, multicol=None, gamma=None, showPhases=False, fontSize=18, linkElements = None):
    """
        lauches a java-based viewer view5d
        Since the viewer is based on calling java via the pyjnius Java bridge, it runs in its own thread each time. This causes overhead and may be not advisable for large
        datasets, but it is nice for debugging purposes as it can also display the data within the pdg debugger.
        -----
        data : multidimensional data to display. This can be up to 5d data
        SX,SY : size of the total viewer in pixels to use (default: 1200,1200)
        mulitcolor : If not None or False, the viewer will be launched in a multicolor (RGB) mode by default. Default: None
        gamma   :  gamma to use for the display. For complex valued magnitude a default of 0.15 is chosen. If not None, a gamma correction is also applied to real valued data.
        showPhases : if True, the phases are shown in multiplicative color overlay with a cyclic colormap
          
        Example:
        import NanoImagingPack as nip
        v=nip.v5(np.random.rand(10,10,10,4),multicol=True)
    """
#    data=np.transpose(data) # force a cast to np.array
    if not isinstance(data,image.image):
        data = image.image(data)

    try:
        import jnius as jn
    except:
        print("unable to import pyjnius. Setting default viewer to a non-java viewer.")
        config.setDefault('IMG_VIEWER', 'NIP_VIEW')
        return vv(data)

    print("Lauching View5D: with datatype: "+str(data.dtype))
    VD = jn.autoclass('view5d.View5D')
#    data=expanddim(data,5) # casts all arrays to 5D
    #    data=np.transpose(data) # reverts all axes to common display direction
#    data=np.moveaxis(data,(4,3,2,1,0),(0,1,2,3,4)) # reverts axes to common display direction
    if not isinstance(data,np.ndarray):
        try:
            import tensorflow as tf
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                data=image.image(data.eval())
            if not isinstance(data,np.ndarray):
                raise ValueError("v5: cannot evaluate tensor.")
        except ImportError:
            raise ValueError("v5: unsupported datatype to display")
    sz=data.shape
    sz=sz[::-1] # reverse
    sz=np.append(sz, np.ones(5-len(data.shape)))

    if (data.dtype=='complex' or data.dtype=='complex128' or data.dtype=='complex64'):
        dcr = data.real.flatten()
        dci = data.imag.flatten()
        #dc=np.concatenate((dcr,dci)).tolist();
        dc = transformations.stack((dcr,dci), axis=1).flatten().tolist()
        #        dc.reverse()
        # dc=data.flatten().tolist();
        out = VD.Start5DViewerC(dc,sz[0],sz[1],sz[2],sz[3],sz[4],SX,SY)
        #        out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");out.ProcessKeyMainWindow("c");
        if gamma is None:
            gamma = 0.15

        # if type(data.unit) is str:
        #     data.unit = [data.unit]
        #
        # if data.unit is None:
        #     data.unit = ["a.u."]
        # if sz[3] > len(data.unit):
        #     data.unit = data.unit + (int(sz[3]) - len(data.unit)) * ["a.u."]

        out.UpdatePanels()
        if showPhases:
            for E in range(int(sz[3])):
                if sz[3]>1:
                    anElem = util.subslice(data, -4, E)
                else:
                    anElem=data
                phases = (np.angle(anElem)+np.pi)/np.pi*180  # in degrees
                # data.unit.append("deg")  # dirty trick
                out.AddElement(phases.flatten().tolist(),sz[0],sz[1],sz[2],sz[3],sz[4])   # add phase information to data
                out.setName(sz[3]+E, "phase")
                out.setUnit(sz[3]+E, "deg")
                out.setMinMaxThresh(sz[3]+E, 0.0, 360.0) # to set the color to the correct values
                v5ProcessKeys(out,'E') # advance to next element to the just added phase-only channel
                v5ProcessKeys(out,12*'c') # toggle color mode 12x to reach the cyclic colormap
                v5ProcessKeys(out,'vVe') # Toggle from additive into multiplicative display
            if sz[3]==1:
                v5ProcessKeys(out,'C') # Back to Multicolor mode
    else:
        dc=data.flatten().tolist()
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
        if multicol is None and (data.colormodel == "RGB"):
            multicol = True
        if multicol is not None and not multicol:
            v5ProcessKeys(out, 'v') # remove first channel from overlay

    if linkElements is not None:
        out.SetElementsLinked(linkElements)
        v5ProcessKeys(out, 'T')  # Adjust all intensities

    if gamma is not None:
        for E in range(int(sz[3])):
            out.SetGamma(E, gamma)

    if (multicol is None or multicol is False) and (sz[3] > 2 and sz[3] < 5):  # reset the view to single colors
        v5ProcessKeys(out,'CGeGveGvEEv') # set first 3 colors to gray scale and remove from color overlay 
    if (multicol is None or multicol is False) and (sz[3] == 2):  # reset the view to single colors
        v5ProcessKeys(out,'CGeGvEv')

    if data.name is not None:
        out.NameWindow(data.name)
    else:
        name = util.caller_args()[0]
        out.NameWindow(name)

    if fontSize is not None:
        out.setFontSize(fontSize)

    if data.pixelsize is not None:
        pixelsize = data.pixelsize
    else:
        pixelsize = 1.0

    pxs = util.expanddimvec(pixelsize, 5, trailing=False)
    pxs = [0.0 if listelem is None else listelem for listelem in pxs]  # replace None values with zero for display
    Names = ['X', 'Y', 'Z', 'E', 'T']
    if (not data.unit is None) and (type(data.unit) == list) and len(data.unit)>4:
        Units = data.unit[0:5]
    else:
        Units = [data.unit,data.unit,data.unit,'ns','s']
    SV = 1.0  # value scale
    NameV = 'intensity'
    UnitV = 'photons'
    out.SetAxisScalesAndUnits(0, SV, pxs[-1], pxs[-2], pxs[-3], pxs[-4], pxs[-5], 0, 0, 0, 0, 0, 0, NameV, Names, UnitV, Units)

    if data.dim_description is not None:
        if type(data.dim_description) == str:
            names = [data.dim_description]
        else:
            names = data.dim_description
        e=0
        for name in names:
            if e < sz[3]: # has to be a legal element
                out.NameElement(e, name)
                e += 1
    else:   # no name was given. Try to recover the caller's name
        name = util.caller_args()[0]
        for e in range(sz[3].astype(int)):
                out.NameElement(e, name)

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