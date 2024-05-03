from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

import subprocess
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = sourcedir

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_EXECUTABLE=' + sys.executable]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)

setup(
    name='mkcfup',
    version='0.1',
    author='SamuraiBUPT',
    author_email='31409163@bupt.edu.cn',
    description='A high speed tracking operator based on MKCFup algorithm.',
    long_description='',
    ext_modules=[CMakeExtension('mkcfup', sourcedir=os.path.join(os.getcwd(), 'csrc'))],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)