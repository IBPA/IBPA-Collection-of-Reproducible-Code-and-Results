# -*- coding: utf-8 -*-
"""setup.py description.

This is a setup.py template for any project.

"""

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='acbm',
    version='1.0.0',
    description='Animal-cell-based meat model sensitivity analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fangzhouli/ACBM-SA',  # GitHub link.
    author='Fangzhou Li',
    author_email='fzli@ucdavis.edu',
    keywords='animal-cell-based meat, sensitivity analysis',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'xlrd>=1.0.0'
        'SALib>=1.3.11',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'plotly>=4.8.1'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        # 'License ::',
    ]
)
