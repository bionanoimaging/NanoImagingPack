# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:45:37 2018

@author: pi96doc
Major rewrite by RH, 20.07.2019 now basing it on pythonbridge rather than pyjnius as before

Javabridge installation Notes under Windows 10:
with the environment named (for example) IP do the following
conda activate IP
conda install openjdk
set "JDK_HOME=%CONDA_PREFIX%\Library
set MSSdk=1
set DISTUTILS_USE_SDK=1
pip install javabridge
pip install python-bioformats
go the the folder
HOME\\.conda\\envs\\NipTest\\Library\\lib
and delete the ext folder
"""
from logging import warning
from pkg_resources import resource_filename
from . import util
from . import transformations
from . import config
from . import image

global JVM_RUNNING
if ~('JVM_RUNNING' in globals()):
    JVM_RUNNING=0

if (JVM_RUNNING==0):
    fn=resource_filename("NanoImagingPack","resources/View5D_.jar")
    try:
        import javabridge
        jars = javabridge.JARS + [fn]
        try:
            import bioformats
            jars = jars + bioformats.JARS
        except:
            pass
        # add_options('-Dswing.aatext=true', '-Dswing.plaf.metal.controlFont=Tahoma', '-Dswing.plaf.metal.userFont=Tahoma', '-Dawt.useSystemAAFontSettings=on')
        javabridge.start_vm(max_heap_size="2G", class_path=jars)  # ,run_headless=True
        javabridge.attach()
#        javabridge.detach()
        JVM_RUNNING = 1
    except Exception as e:
        print(e)
        print("Problem setting classpath. Switching to conventional viewer by setting __DEFAULTS__['IMG_VIEWER'] = 'NIP_VIEW' ")
        config.setDefault('IMG_VIEWER', 'NAPARI') # 'NIP_VIEW'

# jnius_config.add_options('-Xrs', '-Xmx4096')
# jnius_config.set_classpath()
# , 'C:/Users/pi96doc/Documents/Programming/PythonScripts/*'
import numpy as np
#import view 

lastviewer=None
global allviewers
allviewers=[]

# Maybe this would have been the solution to a simple implementation:
# View5D=javabridge.JClassWrapper('view5d.View5D')()

class View5D:
    if JVM_RUNNING:
        setSize = javabridge.make_method("setSize","(II)V")
        setName = javabridge.make_method("setName","(ILjava/lang/String;)V")
        NameElement = javabridge.make_method("NameElement","(ILjava/lang/String;)V")
        NameWindow = javabridge.make_method("NameWindow","(Ljava/lang/String;)V")
        setFontSize = javabridge.make_method("setFontSize","(I)V")
        setUnit = javabridge.make_method("setUnit","(ILjava/lang/String;)V")
        SetGamma = javabridge.make_method("SetGamma","(ID)V")
        setMinMaxThresh = javabridge.make_method("setMinMaxThresh","(IFF)V")
        ProcessKeyMainWindow = javabridge.make_method("ProcessKeyMainWindow","(C)V")
        ProcessKeyElementWindow = javabridge.make_method("ProcessKeyElementWindow","(C)V")
        UpdatePanels = javabridge.make_method("UpdatePanels","()V")
        repaint = javabridge.make_method("repaint","()V")
        hide = javabridge.make_method("hide","()V")
        toFront = javabridge.make_method("toFront","()V")
        SetElementsLinked = javabridge.make_method("SetElementsLinked","(Z)V") # Z means Boolean
        closeAll = javabridge.make_method("closeAll","()V")
        DeleteAllMarkerLists = javabridge.make_method("DeleteAllMarkerLists","()V")
        ExportMarkers = javabridge.make_method("ExportMarkers","(I)[[D")
        ExportMarkerLists = javabridge.make_method("ExportMarkerLists","()[[D")
        ExportMarkersString = javabridge.make_method("ExportMarkers","()Ljava/lang/String;")
        ImportMarkers = javabridge.make_method("ImportMarkers","([[F)V")
        ImportMarkerLists = javabridge.make_method("ImportMarkerLists","([[F)V")
        AddElem = javabridge.make_method("AddElement","([FIIIII)Lview5d/View5D;")
        ReplaceDataB = javabridge.make_method("ReplaceDataB","(I,I[B)V")
        setMinMaxThresh = javabridge.make_method("setMinMaxThresh","(IDD)V")
        SetAxisScalesAndUnits = javabridge.make_method("SetAxisScalesAndUnits","(DDDDDDDDDDDDLjava/lang/String;[Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;)V")

    def ProcessKeys(self, KeyList):
        for k in KeyList:
            self.ProcessKeyMainWindow(k)
            self.UpdatePanels()
            self.repaint()

    def convertDataAndSignature(self, data):
        """
            data: input data
            :return: tuple of (call signature, TypeLetter, converted data and 5D-size)
        """
        env = javabridge.get_env()
        sh = data.shape
        sz = sh[::-1] + (5 - len(sh)) * (1,)  # order and append zeros
        if (data.dtype=='complex' or data.dtype=='complex128' or data.dtype=='complex64'):
            dcr = data.real.flatten(); dci = data.imag.flatten()
            dc = transformations.stack((dcr,dci), axis=1).flatten() # .tolist()
            dc = env.make_float_array(dc.astype('float32'))
            typ = "C"
            sig = "([FIIIII)Lview5d/View5D;"
        else:
            dc=data.flatten() # .tolist()
            if data.dtype == 'float32':
                dc = env.make_float_array(dc)
                typ = "F"
                sig = "([FIIIII)Lview5d/View5D;"
            elif data.dtype == 'float':
                dc = env.make_double_array(dc)
                typ = "D"
                sig = "([DIIIII)Lview5d/View5D;"
            elif data.dtype == 'int':
                dc = env.make_int_array(dc)
                typ = "I"
                sig = "([IIIIII)Lview5d/View5D;"
            elif data.dtype == 'uint8':
                dc = env.make_byte_array(dc)
                typ = "B"
                sig = "([BIIIII)Lview5d/View5D;"
            elif data.dtype == 'bool':
                dc=data.flatten().astype('uint8')
                dc = env.make_byte_array(dc)
                typ = "B"
                sig = "([BIIIII)Lview5d/View5D;"
            elif data.dtype == 'int16':
                dc = env.make_short_array(dc)
                typ = "S"
                sig = "([SIIIII)Lview5d/View5D;"
            elif data.dtype == 'uint16':
                dc = data.flatten().astype("int32")
#                dc = env.make_short_array(dc)
                # klass = env.find_class('[C')
                # dc = env.make_object_array(dc.shape[0], klass)
                # for i in range(dc.shape[0]):
                #     env.set_object_array_element(dc, i, dc[i]);
                typ = "I"
                sig = "([IIIIII)Lview5d/View5D;"
            elif data.dtype == 'uint32' or data.dtype == 'int32':
                dc = env.make_int_array(dc)
                typ = "I"
                sig = "([IIIIII)Lview5d/View5D;"
            else:
                print("View5D: unknown datatype: "+str(data.dtype))
                raise ValueError("View5D: unknown datatype: ")
                return None
        return (sig,typ,dc,sz)

    def AddElement(self, data):
        (sig,typ,dc,sz) = self.convertDataAndSignature(data)
        javabridge.call(self.o, "AddElement", sig, dc, sz[0], sz[1], sz[2], sz[3], sz[4]);
        if data.dtype == 'bool':
            self.ProcessKeys("r")  # color this red by default to indicate that it is a bool

    def ReplaceData(self, data, e=0,t=0,title=None):
        (sig,typ,dc,sz) = self.convertDataAndSignature(data)
        javabridge.call(self.o, "ReplaceData"+typ, "(II"+sig[1:3]+")V",e,t,dc);
        if title is not None:
            if not isinstance(title,str):
                title = str(title)
            self.NameWindow(title)
        self.ProcessKeys("vviZ")
        self.repaint()
        self.toFront()

    def __init__(self, data):
#        javabridge.attach()
        (sig,typ,dc,sz) = self.convertDataAndSignature(data)
        self.o = javabridge.static_call("view5d/View5D", "Start5DViewer"+typ, sig, dc, sz[0], sz[1], sz[2], sz[3], sz[4]);
        self.ProcessKeyMainWindow('i')
        self.ProcessKeys("vv")
        self.toFront()
        if data.dtype == 'bool':
            self.ProcessKeys("r")  # color this red by default to indicate that it is a bool

    def getMarkers(self, ListNo=0, OnlyPos=True):
        env = javabridge.get_env()
        if self != None:
            mJ = self.ExportMarkers(ListNo)
            markerList = env.get_object_array_elements(mJ)
            for n in range(len(markerList)):
                markers = env.get_double_array_elements(markerList[n])
                if OnlyPos:
                    if len(markers) > 0:
                        markers = markers[4::-1]  # extract only position information and orient if for numpy standard
                markerList[n]=markers
        return markerList

    def getMarkerLists(self):
        env = javabridge.get_env()
        if self != None:
            mJ = self.ExportMarkerLists()
            markerList = env.get_object_array_elements(mJ)
            for n in range(len(markerList)):
                markers = env.get_double_array_elements(markerList[n])
                markerList[n]=markers
        return markerList

    def setMarkerLists(self, markers, deleteAll=False):
        env = javabridge.get_env()
        if self != None and len(markers)>0:
            klass = env.find_class('[F')
            jarr = env.make_object_array(len(markers), klass)
            n=0
            for amarker in markers:
                Jm = env.make_float_array(amarker.astype(np.float32))
                env.set_object_array_element(jarr,n,Jm)
                n += 1
            if deleteAll:
                self.DeleteAllMarkerLists()
            self.ImportMarkerLists(jarr)
            self.toFront()
            self.ProcessKeys("jj")

    def setMarkers(self, markers, ListNo=0, OnlyPos=None, scale=1.0, offset=0.0, TagIt=None):
        env = javabridge.get_env()
        if not isinstance(markers,list):
            if not np.isreal(markers):
                markers = np.stack((np.imag(markers),np.real(markers)),0)

            if markers.ndim > 1:
                markers = list(markers)
            else:
                markers = [markers]
        if self != None and len(markers)>0:
            klass = env.find_class('[F')
            jarr = env.make_object_array(len(markers), klass)
            n=0
            for amarker in markers:
                if OnlyPos is None:
                    if amarker.shape[-1] > 5:
                        OnlyPos=False
                    else:
                        OnlyPos=True
                if OnlyPos:
                        mymarker = np.zeros(20,dtype=np.float32)
                        mymarker[0:amarker.shape[-1]] = (amarker * scale + offset)[::-1]
                if TagIt is not None and TagIt is True:
                    mymarker[14]=1.0  # Tags this one
                Jm = env.make_float_array(mymarker)
                env.set_object_array_element(jarr,n,Jm)
                n += 1
            self.ImportMarkers(jarr)
            self.toFront()

    def addMarker(self, pos, listNo=None, markerNo=None, scale=1.0, offset=0.0, TagIt=False, color=None):
        pos = np.array(pos)
        if np.prod(pos.shape) == 1 and not isinstance(pos,np.complex):
            pos = np.stack((np.imag(pos),np.real(pos)),0)
        ML = self.getMarkerLists()
        maxList=0
        for n in range(len(ML)):
            ML[n] = ML[n].astype(np.float32)
            maxList = max(maxList,ML[n][0])
        if listNo is None:
            listNo = maxList
        maxMarker=-1
        for n in range(len(ML)):
            if ML[n][0] == listNo:
                maxMarker = max(maxMarker,ML[n][1])
        if markerNo is None:
            markerNo = maxMarker+1
        # List Nr.,	Marker Nr,	PosX [pixels],	Y [pixels],
        # Z [pixels],	Elements [element],	Time [time], Integral (no BG sub) [Units],
        # Max (no BG sub) [Units],	X [nm],	Y [nm],	Z [nm],
        # E [ns],	T [s],	Integral probMap (no BG sub)[photons],	Max probMap (no BG sub)[photons]
        # TagText 	TagInteger 	Parent1 	Parent2
        # Child1 	 Child2 	ListColor 	ListName
        mymarker = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,-1,-1,-1,-1,0],dtype=np.float32) # np.zeros(22, dtype=np.float32)
        mymarker[2:2+pos.shape[-1]] = (pos * scale + offset)[::-1]
        mymarker[0] = listNo
        mymarker[1] = markerNo
        mymarker[16] = TagIt+0.0
        if color is None:
            color = 0xf0f0f0
        mymarker[21] = (color & ((1 << 31) - 1)) - (color & (1 << 31))  # to compute a color
        ML.append(mymarker)
        # self.setMarkerLists([mymarker]) # res = v.getMarkerLists()
        self.setMarkerLists(ML, deleteAll=True)
        # self.setMarkerLists(ML, deleteAll=True)  # this needs to be done twice for some odd numbering reason

    def getMarkerString(self):
        env = javabridge.get_env()
        return self.ExportMarkersString()

# self.toFront()

def v5ProcessKeys(out,KeyList):
    out.ProcessKeys(KeyList)

def napariAddLayer(data, v=None, gamma=None):
    if gamma != None:
        warning("Removed gamma option from Napari viewer")
    import napari as nap

    if v is None:
        v = nap.Viewer()

    if np.isrealobj(data) or nap.__version__ == '0.3.7rc11.dev148+g4efccaf5':
        v.add_image(data)
        v.dims.set_point(0, data.shape[0] // 2)
        if data.ndim>1:
            v.dims.set_point(1, data.shape[1] // 2)
        if data.ndim>2:
            v.dims.set_point(2, data.shape[2] // 2)
    else:
        v=napariAddLayer(np.imag(data),v)
        v.active_layer.visible = False
        v=napariAddLayer(np.real(data),v)
        v.active_layer.visible = False
        v=napariAddLayer(np.angle(data),v)
        v.active_layer.visible = False
        v=napariAddLayer(np.abs(data),v,0.25)
    return v

def vv(data, SX=1200, SY=1200, multicol=None, gamma=None, showPhases=False, fontSize=18, linkElements = None):
    if config.__DEFAULTS__['IMG_VIEWER'] == 'VIEW5D':
        return v5(data, SX=SX, SY=SY, multicol=multicol, gamma=gamma, showPhases=showPhases, fontSize=fontSize, linkElements = linkElements)
    else:
        ret = data._repr_pretty_([], cycle=False)
        if hasattr(data, 'v') and data.v!=[]:
            return data.v
        else:
            return ret

def v5(data, SX=1200, SY=1200, multicol=None, gamma=None, showPhases=False, fontSize=18, linkElements = None, viewer=None):
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
    if 'Tensor' in str(type(data)):  # check if this may be a tensorflow tensor object
        try:
            import tensorflow as tf
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                data = image.image(data.eval())
            if not isinstance(data,np.ndarray):
                raise ValueError("v5: cannot evaluate tensor.")
        except ImportError:
            raise ValueError("v5: unsupported datatype to display")

    if not isinstance(data, image.image):
        data = image.image(data)

    try:
        import javabridge
    except:
        print("unable to import javabridge. Setting default viewer to a non-java viewer.")
        config.setDefault('IMG_VIEWER', 'NIP_VIEW')
        return vv(data)

    if viewer is None:
        print("Lauching View5D: with datatype: "+str(data.dtype))
    else:
        VD = viewer  # just replace the data
#    data=expanddim(data,5) # casts all arrays to 5D
    #    data=np.transpose(data) # reverts all axes to common display direction
#    data=np.moveaxis(data,(4,3,2,1,0),(0,1,2,3,4)) # reverts axes to common display direction
    sz=data.shape
    sz=sz[::-1] # reverse
    sz=np.append(sz, np.ones(5-len(data.shape)))

    out = View5D(data) # creates the viewer and loads the data
    if (data.dtype=='complex' or data.dtype=='complex128' or data.dtype=='complex64'):
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
                out.AddElement(phases)   # add phase information to data
                out.setName(sz[3]+E, "phase")
                out.setUnit(sz[3]+E, "deg")
                out.setMinMaxThresh(sz[3]+E, 0.0, 360.0) # to set the color to the correct values
                out.ProcessKeys('E') # advance to next element to the just added phase-only channel
                out.ProcessKeys(12*'c') # toggle color mode 12x to reach the cyclic colormap
                out.ProcessKeys('vVe') # Toggle from additive into multiplicative display
            if sz[3]==1:
                out.ProcessKeys('C') # Back to Multicolor mode
    else:
        if multicol is None and (data.colormodel == "RGB"):
            multicol = True
        if multicol is not None and not multicol:
            out.ProcessKeys(out, 'v') # remove first channel from overlay
    out.setSize(SX,SY)

    if linkElements is not None:
        out.SetElementsLinked(linkElements)
        out.ProcessKeys('T')  # Adjust all intensities

    if gamma is not None:
        for E in range(int(sz[3])):
            out.SetGamma(E, gamma)

    if (multicol is None or multicol is False) and (sz[3] > 2 and sz[3] < 5):  # reset the view to single colors
        out.ProcessKeys('CGeGveGvEEv') # set first 3 colors to gray scale and remove from color overlay
    if (multicol is None or multicol is False) and (sz[3] == 2):  # reset the view to single colors
        out.ProcessKeys('CGeGvEv')

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
    pxs = [1.0 if listelem is None else listelem for listelem in pxs]  # replace None values with zero for display
    Names = ['X', 'Y', 'Z', 'E', 'T']
    if (not data.unit is None) and (type(data.unit) == list):
        Units=['a.u.','a.u.','a.u.','a.u.','a.u.']
        if len(data.unit)>4:
            Units = data.unit[-5:]
        else:
            Units[0:len(data.unit)] = data.unit[-1:-len(data.unit)-1:-1]
        Units = ['a.u.' if listelem is None else listelem for listelem in Units]  # replace None values with zero for display
    else:
        Units = [data.unit,data.unit,data.unit,'ns','s']
    SV = 1.0  # value scale
    NameV = 'intensity'
    UnitV = 'photons'
    # the line below set this for all elements and times
    out.SetAxisScalesAndUnits(SV, pxs[-1], pxs[-2], pxs[-3], pxs[-4], pxs[-5], 0, 0, 0, 0, 0, 0, NameV, Names, UnitV, Units)
    # javabridge.static_call("view5d/View5D", "Start5DViewerS", "(IDDDDDDDDDDDDS[SS[S)V", 0, SV, pxs[-1], pxs[-2], pxs[-3], pxs[-4], pxs[-5], 0, 0, 0, 0, 0, 0, NameV, Names, UnitV, Units);

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

    out.ProcessKeys('Ti12') # to initialize the zoom and trigger the display update
#    out.UpdatePanels()
    #out.repaint()
    # out.toFront()
    global allviewers
    allviewers.append(out)
    return out

def v5NameElement(out, enum, Name):
    out.NameElement(enum, Name)

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
    return myviewer.getMarkers(ListNo,OnlyPos)

# if __name__ == '__main__':  # executes the test, when running this file directly
#     import NanoImagingPack as nip
#     obj = nip.readim()
#     vv(obj)

