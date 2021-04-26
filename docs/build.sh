#!/bin/bash

make clean
rm source/autoatlas.rst
sphinx-apidoc -o ./source ../autoatlas #../autoatlas/*sr.py 
make html
