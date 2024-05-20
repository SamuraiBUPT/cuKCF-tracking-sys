#!/bin/bash

rm -rf build
rm cuKCF.cpython-310-x86_64-linux-gnu.so
rm libcuKCF_lib.so
rm -rf cuKCF.egg-info

pip install -e .