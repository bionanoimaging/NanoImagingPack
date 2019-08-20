import NanoImagingPack as nip
# obj = nip.readim()
# nip.v5(obj)

# obj1 = nip.readim('MITO_SIM.tif',z=0)
# obj2 = nip.readim('MITO_SIM.tif',z=0)
# v1=nip.v5(obj2)

obj = nip.readim()
v1 = nip.v5(obj)
v1.AddElement(255.0-obj)  # has a different datatype.
v1.ProcessKeys("e"+25*"cZ") # swapt the second element to a cyclic colormap
import time
for n in range(1,40):
    time.sleep(0.3)
    obj = ((obj + 10) % 256).astype("uint8") + 0.0
    print("Displaying iteration: "+str(n))
    v1.ReplaceData(obj,e=1,title=n)
    # v1.ProcessKeys("Z") # Why has this no effect?

# path = r'C:\Users\pi96doc\Downloads\mitosis.tif'
# obj3 = nip.readim(path,z=1,c=0)
# v2=nip.v5(obj3)

# nip.readim('https://en.wikipedia.org/wiki/Mandrill#/media/File:Mandrill_Albert_September_2015_Zoo_Berlin_(2).jpg')

# import javabridge
#
# rdrClass = javabridge.JClassWrapper('loci.formats.in.OMETiffReader')()
# rdrClass.setOriginalMetadataPopulated(True)
# clsOMEXMLService = javabridge.JClassWrapper('loci.formats.services.OMEXMLService')
# serviceFactory = javabridge.JClassWrapper('loci.common.services.ServiceFactory')()
# service = serviceFactory.getInstance(clsOMEXMLService.klass)
# metadata = service.createOMEXMLMetadata()
# rdrClass.setMetadataStore(metadata)
# rdrClass.setId(path)
# root = metadata.getRoot()
# first_image = root.getImage(0)
# pixels = first_image.getPixels()
# # The plane data isn't in the planes, it's in the tiff data
# for idx in range(pixels.sizeOfTiffDataList()):
#     tiffData = pixels.getTiffData(idx)
#     c = tiffData.getFirstC().getValue().intValue()
#     t = tiffData.getFirstT().getValue().intValue()
#     print("TiffData: c="+c+", t="+t)
