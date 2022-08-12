import setuptools

#  # Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]
#
#  with open("README.md", "r") as fh:
#      long_description = fh.read()
#
#  with open('HISTORY.md') as file:
#      history = file.read()

setuptools.setup(
    name="RVFitter",
    version="0.0.1",
    author="Tim Michel Heinz Wolf",
    description="Organiation of files for RV finding and fitting",
    #  long_description=long_description + '\n\n' + history,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=requires + [
        'pytest',
        'flake8',
    ],
    python_requires='>=3.7',
    url="https://github.com/WolfXeHD/RVFitter",
    packages=setuptools.find_packages(),
    package_data = {"RVFitter": ["tests/test_data"]},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
