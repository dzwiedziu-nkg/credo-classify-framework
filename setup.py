import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="credo_cf",
    version="0.0.1",
    author="Michał Niedźwiecki",
    author_email="nkg753@gmail.com",
    description="CREDO Classification Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dzwiedziu-nkg/credo-classify-framework",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='credo cosmic-ray image classification',
    python_requires='>=3.6, <4',
    install_requires=['pillow', 'numpy', 'pandas', 'matplotlib', 'opencv-python', 'opencv-contrib-python', 'scikit-learn'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/dzwiedziu-nkg/credo-classify-framework/issues',
        'Source': 'https://github.com/dzwiedziu-nkg/credo-classify-framework',
        'Background': 'https://credo.science/',
    },
)
