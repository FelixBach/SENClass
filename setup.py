from setuptools import setup, find_packages
import os
import sys

directory = os.path.abspath(os.path.dirname(__file__))
if sys.version_info >= (3, 0):
    with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
else:
    with open(os.path.join(directory, 'README.md')) as f:
        long_description = f.read()

setup(name='SENClass',
      packages=find_packages(),
      include_package_data=True,
      setup_requires=['setuptools_scm'],
      use_scm_version=True,
      description='Tool for the classification of Sentinel-1 data with a random forest',
      classifiers=[
          'License :: FSF Approved :: MIT License',
          'Operating System :: Microsoft :: Windows',
          'Programming Language :: Python',
      ],
      install_requires=['numpy',
                        'matplotlib',
                        'pandas',
                        'scikit-learn'
                        'gdal',
                        'rasterio',
                        'statsmodels',
                        'setuptools'],
      python_requires='>=3.7.0',
      url='https://github.com/FelixBach/SENClass.git',
      author='Felix Bachmann',  # 'Anastasiia Vynohradova'
      author_email='Felix.Bachmann@uni-jena.de',  # 'anastasiia.vynohradova@uni-jena.de'
      license='MIT',
      zip_safe=False,
      long_description=long_description,
      long_description_content_type='text/markdown')
