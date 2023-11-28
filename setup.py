from setuptools import setup, find_packages

setup(
    name="mbrl",
    description="Code for GCOPFCE, based on facebook's mbrl-lib.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
