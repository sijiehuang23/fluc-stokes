from setuptools import setup, find_packages


setup(
    name='fluc_stokes',
    version=0.1,
    author='Sijie Huang',
    description="A lightweight Fourier solver for fluctuating Stokes equation",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'cupy',
        'cufinufft'
    ]
)
