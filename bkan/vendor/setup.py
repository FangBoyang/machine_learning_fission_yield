from setuptools import setup, find_packages

setup(
    name="svgp_kan",
    version="0.4.4",  # bug fixes
    description="Scalable Probabilistic KANs using Sparse Variational GPs",
    author="Sungtaek Ju",
    url="https://github.com/sungjuGit/svgp-kan",  # Added your repo link
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",      # New
        "Topic :: Scientific/Engineering :: Medical Science Apps.",  # New (U-Net context)
    ],
    keywords=[
        "KAN", 
        "Kolmogorov-Arnold Networks", 
        "Gaussian Processes", 
        "U-Net", 
        "Computer Vision", 
        "Uncertainty Quantification"
    ],
)
