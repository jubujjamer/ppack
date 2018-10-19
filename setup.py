from setuptools import setup

long_description=open('README.md')

install_requires=['scipy', 'numpy', 'scikit-image', 'imageio', 'numba'],

setup(name='phasepack',
                  version='0.1',
                  description='A collection of phase retrieval methods.',
                  url='http://github.com/jubujjamer/phasepack-python/',
                  author='Juan M. Bujjamer',
                  author_email='jubujjamer@df.uba.ar',
                  license='MIT',
                  packages=['phasepack'],
                  # install_requires=install_requires,
                  zip_safe=False)
