import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="cause2e",
    version="0.1",
    author="Daniel Gruenbaum",
    author_email="daniel.gruenbaum@osram-os.com",
    description="A package for end to end causal analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MLResearchAtOSRAM/cause2e",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "dowhy",
        "ipython",
        "networkx",
        "numpy",
        "pandas",
        "pycausal",
        "pydot"
    ]
)
