import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "deepmedic",
    version = "0.5.4",
    author = "Konstantinos Kamnitsas",
    author_email = "konstantinos.kamnitsas12@ic.ac.uk",
    description = "Efficient Multi-Scale 3D Convolutional Neural Network for Brain Lesion Segmentation",
    license = "BSD",
    keywords = "CNN convolutional neural network brain lesion segmentation",
    url = "https://github.com/Kamnitsask/deepmedic",
    download_url = 'https://github.com/Kamnitsask/deepmedic/tarball/0.5.4',
    packages=find_packages(),
    scripts = ['deepMedicRun'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
	"Topic :: Scientific/Engineering :: Artificial Intelligence",
	"Topic :: Scientific/Engineering :: Image Recognition",
	"Topic :: Scientific/Engineering :: Medical Science Apps.",
	"Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
    ],
    install_requires=['nibabel', 'numpy>=1.7.1', 'six>=1.9.0', 'nose>=1.3.0', 'theano>=0.8.0', 'pp'],
    dependency_links=[
	"http://www.parallelpython.com/downloads/pp/pp-1.6.4.tar.gz",
    ]
)
