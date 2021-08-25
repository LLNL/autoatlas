=========
AutoAtlas
=========

-------------
Documentation
-------------
Link to detailed documentation will be provided soon.

--------------------
Installing AutoAtlas
--------------------

* It is recommended to use a python virtual environment such as `anaconda <https://www.anaconda.com/products/individual>`_
* Install pytorch by following the instructions in `pytorch.org <https://pytorch.org/>`_
* It is recommended to install the following python packages from the *conda-forge* respository::
  
     conda -c conda-forge install numpy
     conda -c conda-forge install pyyaml
     conda -c conda-forge install scikit-image
     conda -c conda-forge install scikit-learn
     conda -c conda-forge install matplotlib
 
* Download *autoatlas* from `https://github.com/LLNL/autoatlas <https://github.com/LLNL/autoatlas>`_ and install *autoatlas* by running the following command from within the top level folder *autoatlas*::
  
     pip install .     
   
------------------
Training AutoAtlas
------------------
* Either run the script `aatrain` or use python to import the object `AutoAtlas` from the package `autoatlas.aatlas` and run `AutoAtlas.train`.
* Script `aatrain` is recommended for users without extensive knowledge of `python` and `pytorch`.
* For help on running `aatrain`, run the following command in a terminal::
     
     aatrain -h
* For help on using the object `AutoAtlas` from within python for training, run the following code in the python CLI::

     >>> from autoatlas.aatlas import AutoAtlas
     >>> help(AutoAtlas) #For help on creating AutoAtlas object
     >>> help(AutoAtlas.train) #For help on training AutoAtlas

---------
Inference
---------
* Inference refers to the generation of AutoAtlas partitions and feature embeddings for representation learning. 
* Either run the script `aainfer` or use python to import the object `AutoAtlas` from the package `autoatlas.aatlas` and run `AutoAtlas.process`.
* Script `aainfer` is recommended for users without extensive knowledge of `python` and `pytorch`.
* For help on running `aainfer`, run the following command in a terminal::
     
     aainfer -h
* For help on using the object `AutoAtlas` from within python for training, run the following code in the python CLI::

     >>> from autoatlas.aatlas import AutoAtlas
     >>> help(AutoAtlas) #For help on creating AutoAtlas object
     >>> help(AutoAtlas.process) #For help on training AutoAtlas

-------
LICENSE
-------
AutoAtlas is distributed under the terms of the MIT license.

LLNL-CODE-802877

