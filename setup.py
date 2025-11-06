from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform
from pathlib import Path

__version__ = '0.0.7'

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

ext_modules = [
    Pybind11Extension(
        'nanoflann_ext',
        ['src/lpqnanoflann.cpp',
         'nanoflann/include/nanoflann.cpp',
         'nanoflann/include/lpq.cpp',
         'nanoflann/include/lpq_metric.cpp',
         'nanoflann/include/lpq_l1_nd.cpp',
         'nanoflann/include/lpq_l2_nd.cpp',
         'nanoflann/include/lpq_l12_2d.cpp',
         'nanoflann/include/lpq_l12_3d.cpp',
         'nanoflann/include/lpq_l12_4d.cpp',
         'nanoflann/include/lpq_l21_2d.cpp',
         'nanoflann/include/lpq_l21_3d.cpp',
         'nanoflann/include/lpq_l21_4d.cpp',
         'nanoflann/include/lpq_lp_nd.cpp',
         'nanoflann/include/lpq_lpq_mnd.cpp'],
        include_dirs=["nanoflann/include"],
        cxx_std=17,
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        arch = platform.machine().lower()
        is_x86 = arch in ("x86_64", "amd64", "i386", "i686")

        for ext in self.extensions:
            opts = ["-ffast-math"]

            if is_x86:
                opts += ["-msse2", "-mfpmath=sse", "-march=native"]

            ext.extra_compile_args = opts

        super().build_extensions()

setup(
    name='lpqtree',
    version=__version__,
    author='Etienne St-Onge',
    author_email='Firstname.Lastname@usherbrooke.ca',
    url='https://github.com/StongeEtienne/lpqtree',
    description='Lpq KD Tree, adapted from Nanoflann with Python wrapper',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    license='BSD 2-Clause',
    packages=['lpqtree'],
    install_requires=['pybind11>=2.8', 'scikit-learn>=1.2', 'numpy>=1.22.3', 'scipy>=1.3.2'],
    setup_requires=['pybind11>=2.8'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
