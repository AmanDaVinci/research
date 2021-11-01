#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
from setuptools import setup

from image_classification import __version__

NAME = 'image-classification'
DESCRIPTION = 'A deep learning project for classification of fashion images.'
URL = 'https://github.com/amandavinci/research'
EMAIL = 'amanthevinci@gmail.com'
AUTHOR = 'Aman'
REQUIRES_PYTHON = '>=3.7.5'
REQUIRED = [
    'hydra-core==1.0',
    'wandb==0.10.10',
    'numpy==1.21.3',
    'torch==1.10.0',
    'torchvision==0.11.1',
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['image_classification'],
    package_data={'image-classifier': [
        'model_archive/model-250240-0.292.pt'
    ]},
    # py_modules=['mypackage'],
    entry_points={
        'console_scripts': [
            'train_image_classification=image_classification.train:main'
            ],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
)