from setuptools import find_packages,setup
from typing import List

def get_requirements(filepath:str)->List[str]:
    requirements=[]
    with open(filepath) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if '-e .' in requirements:
            requirements.remove("-e .")
        return requirements
    
setup(
    name="MLproject",
    version="0.0.1",
    author="chetan",
    author_email="chetanmaini2@gmail.com",
    packages=find_packages(),
    install_packages=get_requirements('requirements.txt')
)