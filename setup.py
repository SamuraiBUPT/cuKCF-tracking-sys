# By SamuraiBUPT

# NOTICE: you should install this python package in a linux environment with CUDA installed.

import os
import re
import subprocess
import sys
import setuptools
import platform
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-Dpybind11_DIR=' + pybind11_cmake_dir,
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() != "Windows":
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='cuKCF',
    version='0.1',
    author='Yazhu Dong',
    author_email='31409163@bupt.edu.cn',
    description='CUDA operator for FHOG computing, aiming at optimizing the inference speed of KCF tracker.',
    long_description='',
    ext_modules=[CMakeExtension('cuKCF', 'csrc')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['pybind11'],
)
