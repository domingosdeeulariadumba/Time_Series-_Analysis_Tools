from setuptools import setup, find_packages

setup(
    name='time_series',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'statsmodels>=0.12.0'
    ],
    
    author='Domingos de Eulária Dumba',
    author_email='domingosdeeulariadumba@gmail.com',
    description = 'A package for time series analysis',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
