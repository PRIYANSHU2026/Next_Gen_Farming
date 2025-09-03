from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="soil_quality_fertility_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Priyanshu Tiwari",
    author_email="example@example.com",
    description="A system for soil quality and fertility prediction using IoT sensors and ML",
    keywords="soil, agriculture, iot, machine learning",
    python_requires=">=3.8",
)