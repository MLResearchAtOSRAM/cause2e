import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="cause2e",
    version="0.2.0",
    author="Daniel Gruenbaum",
    author_email="daniel.gruenbaum@ams-osram.com",
    description="A package for end-to-end causal analysis",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MLResearchAtOSRAM/cause2e",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows"
    ],
    python_requires='>=3.7',
    install_requires=[
        "dowhy",
        "graphviz",
        "ipython",
        "jinja2",
        "networkx",
        "numpy",
        "pandas",
        "pillow",
        "pyarrow",
        "pycausal",
        "pydot",
        "seaborn"
    ]
)
