import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="frxas",
    version="0.0.1",
    author="frXAS.py developers",
    author_email="brian.s.gerwe@gmail.com",
    description="A Python package to work with data from frequency-resolved \
                X-ray absorption spectroscopy measurements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://frxaspy.readthedocs.io/en/latest/",
    packages=setuptools.find_packages(),
    install_requires=['matplotlib>=3.0', 'numpy>=1.14', 'scipy>=1.0'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
    ],
)
