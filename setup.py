import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os

if not("LAL_PREFIX" in os.environ):
    print("No LAL installation found, please install LAL from source or source your LAL installation")
    exit()
else:
    lal_prefix = os.environ.get("LAL_PREFIX")

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())
        
    
lal_includes = lal_prefix+"/include"
lal_libs = lal_prefix+"/lib"

ext_modules=[
             Extension("cosmolisa.cosmology",
                       sources=["cosmolisa/cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),lal_includes,"cosmolisa"]
                       ),
             Extension("cosmolisa.likelihood",
                       sources=["cosmolisa/likelihood.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),lal_includes,"cosmolisa"]
                       )
             ]

setup(
      name = "cosmolisa",
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include(),lal_includes,"cosmolisa/cosmolisa"],
      description='cosmolisa: a cpnest model for cosmological inference with LISA',
      author='Walter Del Pozzo, Danny Laghi',
      author_email='walter.delpozzo@ligo.org, danny.laghi@ligo.org',
      url='https://github.com/wdpozzo/cosmolisa',
      license='MIT',
      cmdclass={'build_ext': build_ext},
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'],
      keywords='lisa gravitational waves cosmology bayesian inference',
      packages=['cosmolisa'],
      install_requires=['numpy', 'scipy', 'corner', 'cython'],
      package_data={"": ['*.pyx', '*.pxd']},
      )

