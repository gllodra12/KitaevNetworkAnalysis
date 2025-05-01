from setuptools import setup, find_packages

setup(
    name="kitaev-network-analysis",
    version="1.0.0",
    description="Complex network analysis tools for the Kitaev model",
    packages=find_packages(),
    author="Guillem Llodrà",
    author_email="gllodra1225@gmail.com",
    python_requires='>=3.8',
    install_requires=[
        'qutip',
        'openfermion',
        'networkx',
        'matplotlib'
    ],
    include_package_data=True,
    license='MIT',
)