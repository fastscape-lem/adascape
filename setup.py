from setuptools import setup


setup(name='adascape',
      version='1.0.0',
      description='Simple adaptive speciation model as a FastScape-LEM component',
      url='https://github.com/fastscape-lem/adascape.git',
      author='Esteban Acevedo-Trejos, Katherine Kravitz, Benoit Bovy',
      author_email='esteban.acevedo-trejos@gfz-potsdam.de, kravitz@gfz-potsdam.de, bbovy@gfz-potsdam.de',
      license='GPLv3',
      packages=['adascape'],
      install_requires=['numpy', 'scipy', 'pandas', 'dendropy', 'fastscape', 'orographic_precipitation'],
      zip_safe=False)
