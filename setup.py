import os
import sys
from setuptools import setup

if sys.version_info[0] != 3:
    raise RuntimeError('Unsupported python version "{0}"'.format(
        sys.version_info[0]))

def _get_file_content(file_name):
    with open(file_name, 'r') as file_handler:
        return str(file_handler.read())
def get_long_description():
    return _get_file_content('README.md')

#on_rtd = os.environ.get('READTHEDOCS') == 'True'

#if not on_rtd:
#    INSTALL_REQUIRES = [
#        'pandas',
#        'requests',
#    ]
#else:
#    INSTALL_REQUIRES = [
#        'requests',
#    ]

INSTALL_REQUIRES = [
    'transformers==2.6.0',
    'simpletransformers==0.22.1',
    'pandas',
    'torch',
    'torchvision',
    'tensorboardX'
]

setup(
    name="smaberta",
    version='0.0.1',
    author="Vishakh Padmakumar, Zhanna Terechshenko",
    description="a wrapper for the huggingface transformer libraries",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords='nlp transformers classification text-classification fine-tuning',
    url="https://github.com/SMAPPNYU/SMaBERTa.git",
    packages=['smaberta'],
    py_modules=['smaberta'],
    license="MIT",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=INSTALL_REQUIRES
)
