import os
import scyjava as sj
import imagej
# create imagej session
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.9.10-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

ij = imagej.init('sc.fiji:fiji', mode='interactive')

def impl_version(fqcn: str):
    c = sj.jimport(fqcn)
    # get the underlying java.lang.Class
    jclass = c.class_
    pkg = jclass.getPackage()
    return None if pkg is None else pkg.getImplementationVersion()

print("ImageJ2:", impl_version("net.imagej.ImageJ"))
print("SCIFIO:",  impl_version("io.scif.SCIFIO"))
print("Bio-Formats core:", impl_version("loci.formats.FormatTools"))
print("Bio-Formats plugins:", impl_version("loci.plugins.BF"))