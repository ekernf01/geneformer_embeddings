from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description = fh.read()

setup(
    name='geneformer_embeddings',
    version='0.0.1',
    description='Extract cell embeddings from GeneFormer',
    long_description=long_description,
	long_description_content_type='text/markdown',
    #url
    author='Eric Kernfeld',
    author_email='eric.kern13@gmail.com',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "anndata",
        "biomart",
        "transformers",
    ],
    python_requires=">=3.7", 
    url='https://github.com/ekernf01/geneformer_embeddings',
)