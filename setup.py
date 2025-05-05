from setuptools import setup, find_packages
import os
import re

# Read version from latentsync/__init__.py
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "latentsync", "__init__.py")
    try:
        with open(version_file, "r") as f:
            content = f.read()
            # Match __version__ = "x.y.z" or __version__ = 'x.y.z'
            version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
            raise RuntimeError("Unable to find version string in %s" % version_file)
    except Exception as e:
        raise RuntimeError("Failed to read version from %s: %s" % (version_file, str(e)))

# Read requirements from requirements.txt
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(req_file, "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            return requirements
    except Exception as e:
        raise RuntimeError("Failed to read requirements from %s: %s" % (req_file, str(e)))

setup(
    name="latentsync",
    version=read_version(),
    description="Audio Conditioned Latent Diffusion Models for Lip Sync",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pinch",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["latentsync*"]),
    package_data={
        "latentsync": ["**/*"]
    },
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "latentsync-infer=latentsync.inference.face_infer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    url="https://github.com/pinch-eng/latentsync",
    project_urls={
        "Homepage": "https://github.com/pinch-eng/latentsync",
        "Repository": "https://github.com/pinch-eng/latentsync.git",
    },
)