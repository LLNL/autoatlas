from setuptools import setup,find_packages
  
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='autoatlas',
      version='0.1.0',
      description='Python package implementing AutoAtlas',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://lc.llnl.gov/bitbucket/projects/REAL/repos/autoatlas',
      author='K. Aditya Mohan',
      author_email='mohan3@llnl.gov',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy','pyyaml','scikit-image','scikit-learn','matplotlib','torch'],
      keywords='AutoAtlas,MRI,Representation Learning',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      entry_points = {'console_scripts': [ 
                'aatrain = aascripts.aatrain:main',
                'aainfer = aascripts.aainfer:main',
                'aaloss = aascripts.aaloss:main',
                'aacompare = aascripts.aacompare:main',
                'aarlearn = aascripts.aarlearn:main',
                'drtrain = aascripts.drtrain:main', 
                'drinfer = aascripts.drinfer:main' 
                ]}
      )
