import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
             Extension("cosmology",
                       sources=["cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = ["/Users/wdp/opt/master/lib"],
                       include_dirs=[numpy.get_include(),"/Users/wdp/opt/master/include"]
                       )
             ]

setup(
      name = "cosmology",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),"/Users/wdp/opt/master/include"]
      )
ext_modules=[
             Extension("likelihood",
                       sources=["likelihood.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = ["/Users/wdp/opt/master/lib"],
                       include_dirs=[numpy.get_include(),"/Users/wdp/opt/master/include"]
                       )
             ]

setup(
      name = "likelihood",
      ext_modules = cythonize(ext_modules),
      include_dirs=[numpy.get_include(),"/Users/wdp/opt/master/include"]
      )

