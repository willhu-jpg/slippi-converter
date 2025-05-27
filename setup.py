from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="slippi_converter",
        version="0.0.1",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        description="A converter for Slippi replay files to a format usable by the reinforcement learning environment",
        python_requires=">=3.7",
    )