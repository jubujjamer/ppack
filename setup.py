from setuptools import setup

long_description=open('README.md')

# install_requires=['scipy', 'numpy', 'scikit-image', 'imageio', 'numba'],

try:
    import numpy
except:
    raise ImportError('Module numpy is not installed.\n\
    Please run \'conda install numpy\' from a conda environment.')

try:
    import scipy
except:
    raise ImportError('Module scipy is not installed.\n\
    Please run \'conda install scipy\' from a conda environment.')

install_requires=['scikit-image', 'imageio', 'numba'],
setup(name='ppack',
        version='0.1',
        description='A collection of phase retrieval methods.',
        url='http://github.com/jubujjamer/ppack/',
        author='Juan M. Bujjamer',
        author_email='jubujjamer@df.uba.ar',
        license='MIT',
        packages=['ppack'],
        install_requires=install_requires,
        zip_safe=False)
