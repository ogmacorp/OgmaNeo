# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from distutils.command.build import build as build
from distutils.command.build_ext import build_ext
from distutils.command.install import install
from distutils import sysconfig

import shutil
import shlex
import subprocess
import os, sys
import os.path
from os import chdir, getcwd
from os.path import abspath, dirname, split

# Check if we're running 64-bit Python
is64bit = sys.maxsize > 2**32

# Check if this is a debug build of Python.
if hasattr(sys, 'gettotalrefcount'):
    build_type = 'Debug'
else:
    build_type = 'Release'


# Subclass the distutils install command
class install_subclass(install):
    description = "Building the OgmaNeo C++ library, generating SWiG bindings, and installing neo"

    def run(self):
        # Run build_ext first, so that it can generate
        # the OgmaNeo library and SwiG the neo.i file
        self.run_command("build_ext")

        # Return through the usual super().run
        return install.run(self)


# Subclass the disutils build command, to reorder sub_commands
# (build_ext *before* build_py)
class build_subclass(build):
    sub_commands = [('build_ext',     build.has_ext_modules),
                    ('build_py',      build.has_pure_modules),
                    ('build_clib',    build.has_c_libraries),
                    ('build_scripts', build.has_scripts),
                   ]


# Subclass the distutils build_ext command
class build_ext_subclass(build_ext):
    description = "Building the C-extension for PyOgmaNeo with CMake"
    user_options = [('extra-cmake-args=', None, 'extra arguments for CMake')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.extra_cmake_args = ''

    def get_ext_path(self, name):
        build_py = self.get_finalized_command('build_py')
        package_dir = build_py.get_package_dir('ogmaneo')
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if suffix is None:
            suffix = sysconfig.get_config_var('SO')
        suffix = "." + suffix.rsplit(".", 1)[-1]
        filename = "../" + name + suffix
        return os.path.abspath(os.path.join(package_dir, filename))

    def get_ext_name(self, name):
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if suffix is None:
            suffix = sysconfig.get_config_var('SO')
        suffix = "." + suffix.rsplit(".", 1)[-1]
        return name + suffix

    def get_py_path(self, name):
        build_py = self.get_finalized_command('build_py')
        package_dir = build_py.get_package_dir('ogmaneo')
        filename = "../" + name + ".py"
        return os.path.abspath(os.path.join(package_dir, filename))

    def get_py_name(self, name):
        return name + ".py"

    def build_extensions(self):
        global build_type

        # The directory containing this setup.py
        source = dirname(abspath(__file__))

        # The staging directory for the library being built
        build_temp = os.path.join(os.getcwd(), self.build_temp)
        build_lib = os.path.join(os.getcwd(), self.build_lib)

        # Change to the build directory
        saved_cwd = getcwd()
        if not os.path.isdir(build_temp):
            self.mkpath(build_temp)
        chdir(build_temp)

        extra_cmake_args = shlex.split(self.extra_cmake_args)
        cmake_command = ['cmake'] + extra_cmake_args

        if "-G" not in self.extra_cmake_args:
            cmake_generator = 'Unix Makefiles'

            if sys.platform == 'darwin':
                cmake_generator = 'Xcode'

            elif sys.platform == 'win32':
                if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor <= 2):
                    cmake_generator = 'MinGW Makefiles'
                else:
                    if sys.version_info.major == 3 and (sys.version_info.minor == 3 or sys.version_info.minor == 4):
                        cmake_generator = 'Visual Studio 10 2010'
                    else:
                        cmake_generator = 'Visual Studio 14 2015'
                    if is64bit:
                        cmake_generator += ' Win64'

            cmake_command += ['-G', cmake_generator]
            cmake_command += ['-DPYTHON_VERSION='+str(sys.version_info.major)]

        cmake_command.append(source)
        subprocess.call(cmake_command)

        if sys.platform == 'win32' or sys.platform == 'darwin':
            self.spawn(['cmake', '--build', '.', '--target', 'install', '--config', build_type])
        else:
            self.spawn(['cmake', '--build', '.', '--config', build_type])

        if not self.inplace:
            # Move the library and neo.py bindings interface
            # to the place expected by the Python build
            self._found_names = []
            built_ext = self.get_ext_name("_ogmaneo")
            if os.path.exists(built_ext):
                ext_path = os.path.join(build_lib, built_ext)
                if os.path.exists(ext_path):
                    os.remove(ext_path)
                self.mkpath(os.path.dirname(ext_path))
                print('Moving library', built_ext,
                      'to build path', ext_path)
                shutil.move(built_ext, ext_path)
                self._found_names.append("_ogmaneo")

                built_py = self.get_py_name("ogmaneo")
                py_path = os.path.join(build_lib, built_py)
                print('Moving Py file', built_py,
                      'to build path', py_path)
                shutil.move(built_py, py_path)
            else:
                raise RuntimeError('C-extension failed to build:',
                                   os.path.abspath(built_ext))

        chdir(saved_cwd)

    def get_names(self):
        return self._found_names

    def get_outputs(self):
        # Just the C extensions
        return [self.get_ext_path(name) for name in self.get_names()]


extension_mod = Extension(
    name="_ogmaneo",
    sources=["ogmaneo.i"]
)

setup(
    name="ogmaneo",
    version="1.4.2",
    description="Python bindings for OgmaNeo library",
    long_description='https://github.com/ogmacorp/PyOgmaNeo',
    author='Ogma Intelligent Systems Corp',
    author_email='info@ogmacorp.com',
    url='https://ogmacorp.com/',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    py_modules=["ogmaneo"],
    ext_modules=[extension_mod],
    cmdclass={
        'build': build_subclass,
        'build_ext': build_ext_subclass,
        'install': install_subclass
    },
)
