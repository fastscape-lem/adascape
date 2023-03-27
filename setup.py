from setuptools import setup


setup(name='adascape',
      version='0.1.0b6',
      description='Simple adaptive speciation models as a landscape evolution model component',
      url='https://git.gfz-potsdam.de/sec55-public/adaptive-speciation',
      author='Esteban Acevedo-Trejos, Katherine Kravitz, Benoit Bovy',
      author_email='esteban.acevedo-trejos@gfz-potsdam.de, kravitz@gfz-potsdam.de, bbovy@gfz-potsdam.de',
      license='GPLv3',
      packages=['adascape'],
      install_requires=['numpy', 'scipy', 'pandas', 'dendropy', 'orographic_precipitation'],
      zip_safe=False)
