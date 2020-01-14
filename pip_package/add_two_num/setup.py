#!/usr/bin/env python
#-*- coding:utf-8 -*-
 
#############################################
# File Name: setup.py
# Author: hezd
# Mail: w-s-h-z-d@163.com
# Created Time: 2020-1-14 19:37:34
#############################################
 
 
from setuptools import setup, find_packages
 
setup(
  name = "two_num_sum",
  version = "0.1",
  license = "MIT Licence",
 
  url = "https://github.com/wshzd/NLP/new/master/pip_package/two_num_sum",
  author = "hezd",
  author_email = "w-s-h-z-d@163.com",
 
  packages = find_packages(),
  include_package_data = True,
  platforms = "any",
  install_requires = []
)
