from setuptools import setup, find_packages

setup(
    name="solar-capstone",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'pymongo',
        'sentence-transformers',
        'PyPDF2',
        'python-docx',
        'beautifulsoup4',
        'python-pptx',
        'openpyxl',
        'lxml',
    ],
    python_requires='>=3.8',
)