from setuptools import setup, find_packages

setup(
    name='omp_impact_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
    ],
    entry_points={
        'console_scripts': [
            'omp_impact_recommender=omp_impact_recommender.cli_tool:main',
        ],
    },
    include_package_data=True,
)