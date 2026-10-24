from distutils.core import setup 
from Cython.Build import cythonize

setup(
        name = "Flocking Confluent Marchetti",
        ext_modules = cythonize('*.pyx', annotate=True),
)
