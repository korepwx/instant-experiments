#!/usr/bin/env python
import ast
import codecs
import re

from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with codecs.open('ipwxlearn/__init__.py', 'rb', 'utf-8') as f:
    version = str(ast.literal_eval(_version_re.search(f.read()).group(1)))

with codecs.open('requirements.txt', 'rb', 'utf-8') as f:
    install_requires = f.readlines()

setup(
    name='ipwxlearn',
    version=version,
    description='Machine learning codebase of Haowen Xu',
    author='Haowen Xu',
    author_email='public@korepwx.com',
    url='https://git.peidan.me/xhw15/ipwxlearn',
    packages=find_packages(),
    platforms='any',
    setup_requires=['setuptools'],
    install_requires=install_requires
)
