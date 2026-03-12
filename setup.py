from setuptools import setup, Extension

# Defining the extension module
module = Extension(
    "symnmfmodule", 
    sources=["symnmf.c", "symnmfmodule.c"], # Compiles both files together
    libraries=['m']
)

setup(
    name='symnmfmodule',
    version='1.0',
    description='Python wrapper for custom C extension',
    ext_modules=[module]
)