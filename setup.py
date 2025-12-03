# setup.py
from setuptools import setup, find_packages

setup(
    name="maze_rl",
    version="1.0.0",
    description="Reinforcement Learning in 2D Maze Environments",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy<2.0",
        "gymnasium>=0.29.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "numba>=0.57.0",
        "scikit-image>=0.21.0",
        "pygame>=2.5.0"
    ],
    python_requires=">=3.8",
)