"""A setuptools based setup module for the py_dic package."""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='py_dic',
    version='1.0.0',
    description='A DIgital Image Correlation implementation in Python using the SciPy stack.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ladisk/pyDIC',
    author='Domen Gorjup',
    author_email='domen_gorjup@hotmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Computer Vision',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='computer vision dic',
    packages=[],#find_packages(exclude=['py_dic.main']),
    py_modules = ['py_dic.dic', 'py_dic.dic_tools', 'py_dic.scheduled_run'],

    install_requires=['matplotlib>=2.0.0',
                      'numpy>=1.14.0',
                      'tqdm>=4.10.0'],
)