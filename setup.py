from setuptools import setup

long_description=open('README.md')

install_requires=['scipy', 'numpy', 'scikit-image', 'imageio', 'numba'],

setup(name='ppack',
                  version='0.1',
                  description='A collection of phase retrieval methods.',
                  url='http://github.com/jubujjamer/phasepack-python/',
                  author='Juan M. Bujjamer',
                  author_email='jubujjamer@df.uba.ar',
                  license='MIT',
                  packages=['ppack'],
                  # install_requires=install_requires,
                  zip_safe=False)
