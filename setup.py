"""A setuptools based setup module for the py_dic package."""

from setuptools import setup, find_packages
from os import path


def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')

with open('README.md', 'r', encoding='UTF-8') as f:
    readme = f.read()


setup(
    name='py_dic',
    version='1.0.0',
    description='A DIgital Image Correlation implementation in Python using the SciPy stack.',
    long_description=readme,
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
    packages=['py_dic'],
    install_requires=requirements,
)