"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='assign_enc',
        version='1.0',
        description='Assignment Encoding',
        author='Jasper Bussemaker',
        author_email='jasper.bussemaker@dlr.de',
        classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
        ],
        license='MIT',
        install_requires=[
            'appdirs',
            'numpy',
            'pandas<3.0',
            'scipy>=1.9.0',
            'numba~=0.56',
        ],
        python_requires='>=3.7',
        packages=find_packages(exclude=['tests*']),
    )
