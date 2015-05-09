from setuptools import setup
from setuptools import find_packages

setup(name='Keras',
      version='0.0.1',
      description='Theano-based Deep Learning',
      author='Kaisheng Yao',
      author_email='kaishengy@gmail.com',
      url='https://github.com/kaishengyao/keras',
      license='MIT',
      install_requires=['theano', 'h5py'],
      packages=find_packages(),
)
