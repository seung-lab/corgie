import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corgie",
    version="0.1.1",
    author="Sergiy Popovych",
    author_email="sergiy.popovich@gmail.com",
    description="Connectomics Registration General Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supersergiy/corgie",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
      'torchfields',
      'torch>=1.8',
      'gevent>=21.1.2',
      'torchvision>=0.9.1',
      'numpy>=1.21',
      'six>=1.16',
      'pyyaml>=5.4.1',
      'mazepa>=0.1.1',
      'modelhouse>=0.1.2',
      'click-option-group>=0.5',
      'click>=7,<8',
      'procspec>=0.0.4',
      'metroem>=0.1.1',
      'idna>=2.5',
      'google-auth>=1.11.0',
      'scikit-image>=0.18',
      'h5py>=3.3.0',
      'kimimaro>=2.1.0',
      'docutils<0.18',
      'cloud-files>=2.0.1',
      'cloud-volume>=5.0.0',
      'sphinx-click>=3',
      'sphinxcontrib-spelling',
      'sphinx_rtd_theme',
    ],
    entry_points={
        "console_scripts": [
            "corgie = corgie.main:cli",
            "corgie-worker = corgie.worker:worker",
        ],
    },
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
