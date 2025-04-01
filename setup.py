from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="entities",
    version="0.1.0",
    author="Francis N.",
    author_email="francis.neequaye@projectdavid.co.uk",
    description="SDK for managing AI entities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frankie336/entitites_sdk",
    package_dir={"": "src"},  # Tells setuptools that packages live under src/
    packages=find_packages(where="src", include=["entities", "entities.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anyio", "certifi", "h11", "httpcore", "httpx", "idna", "sniffio",
        "fastapi", "databases", "uvicorn", "sqlalchemy",
        "pydantic", "starlette", "asgiref", "click", "pymysql", "cryptography",
        "typing_extensions", "python-dotenv",'sentence_transformers'
        'validators', 'pdfplumber', 'asyncio'
    ],
    extras_require={
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "entities-api=src.entities.main:main",
        ],
    },
)
