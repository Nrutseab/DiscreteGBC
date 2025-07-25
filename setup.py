from setuptools import setup, find_packages

setup(
    name='discrete_gbc',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.5',
        'h5py>=3.7',
        'pandas>=1.3',
        'scikit-learn>=1.0',
        'networkx>=2.6'
    ],
    entry_points={
        'console_scripts': [
            'detect-echoes=DiscreteGBC.LIGO_TetrahedralEchoes.detect_echoes:main'
        ]
    },
)
