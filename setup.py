from setuptools import find_packages,setup

setup(
    name="MCQgenerator",
    version="0.0.1",
    author="Sintayehu Bikila",
    author_email="sench729@gmail.com",
    install_requires=["openai", "langchain", "streamlit", "python-dotenv",
                      "PyPDF2"],
    packages=find_packages()
)