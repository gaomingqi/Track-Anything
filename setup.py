from setuptools import setup, find_packages

setup(
    name="tracker",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[],
    author="zhaifang",
    author_email="zhaifang@tsinghua.edu.cn",
    description="xmem tracking for 3 sensor short long-term memory",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="git@github.com:bingxinhu/Track-Anything.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
