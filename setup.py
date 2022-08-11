from setuptools import setup


setup(name='adascape',
      version='0.1.0',
      description='A simple Parapatric Speciation Model as a LEM component',
      url='https://gitext.gfz-potsdam.de/sec55-public/parapatric-speciation',
      author='Katherine Kravitz, Benoit Bovy',
      author_email='kravitz@gfz-potsdam.de, bbovy@gfz-potsdam.de',
      license='GPLv3',
      packages=['adascape'],
      install_requires=['numpy', 'scipy', 'pandas'],
      zip_safe=False)
