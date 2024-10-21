from setuptools import setup, find_packages
import io
import os
from pathlib import Path


# metadata of package
PACKAGE_NAME = 'prediction_model'
DESCRIPTION = 'A simple package'
URL = 'https://github.com/e-espootin/'
EMAIL = 'E.Espootin@gmail.com'
Author = 'Ebrahim Espootin'
Requires_Python = '>=3.7.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# get the lis of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# load the package's __version__ module as a dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / PACKAGE_NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=PACKAGE_NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=Author,
    author_email=EMAIL,
    python_requires=Requires_Python,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={PACKAGE_NAME: ['VERSION']},
    include_package_data=True,
    license='MIT',
    install_requires=list_reqs(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
