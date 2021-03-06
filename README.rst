=========
AutoAtlas
=========

---------
Reference
---------
* K\. A. Mohan, A. D. Kaplan, "AutoAtlas: Neural Network for 3D Unsupervised Partitioning and Representation Learning", *submitted to IEEE Journal of Biomedical and Health Informatics*, 2021 `(link to pdf) <https://arxiv.org/pdf/2010.15987.pdf>`_.

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
 
* Download *autoatlas* from `https://github.com/LLNL/autoatlas <https://github.com/LLNL/autoatlas>`_ and install *autoatlas* by running the following command in a terminal from within the top level folder *autoatlas*::
  
     pip install .     
   
------------------
Training AutoAtlas
------------------
* Either run the script `aatrain` or use python to import the object `AutoAtlas` from the package `autoatlas.aatlas` and run the method `AutoAtlas.train`.
* Script `aatrain` is recommended for users without extensive knowledge of `python` and `pytorch`.
* For help on running `aatrain`, run the following command in a terminal::
     
     aatrain -h
* For help on using the method `AutoAtlas.train` from within python for training, run the following code in the python CLI::

     >>> from autoatlas.aatlas import AutoAtlas
     >>> help(AutoAtlas) #For help on creating AutoAtlas object
     >>> help(AutoAtlas.train) #For help on training AutoAtlas

-----------------------
Inference for AutoAtlas
-----------------------
* Inference refers to the generation of AutoAtlas partitions and feature embeddings for representation learning. 
* Either run the script `aainfer` or use python to import the object `AutoAtlas` from the package `autoatlas.aatlas` and run the method `AutoAtlas.process`.
* Script `aainfer` is recommended for users without extensive knowledge of `python` and `pytorch`.
* For help on running `aainfer`, run the following command in a terminal::
     
     aainfer -h
* For help on using the method `AutoAtlas.process` from within python for inference, run the following code in the python CLI::

     >>> from autoatlas.aatlas import AutoAtlas
     >>> help(AutoAtlas) #For help on creating AutoAtlas object
     >>> help(AutoAtlas.process) #For help on running inference on AutoAtlas

-----------------------------
Learning from Representations
-----------------------------
* Prediction of meta-data from respresentations, i.e., feature embeddings at the bottleneck layer of autoencoders, which was generated by AutoAtlas in the inference step.
* Either execute the script `aarlearn` or use python to import the object `Predictor` from the package `autoatlas.rlearn` and run its methods.
* Script `aarlearn` is recommended for users without extensive knowledge of `python` and `scikit-learn`.
* For help on running `aarlearn`, run the following command in a terminal::

     aarlearn -h
* For help on using the object `Predictor` from within python for predictions, run the following code in the python CLI::

     >>> from autoatlas.rlearn import Predictor
     >>> help(Predictor) #For help on the Predictor object and its methods
     >>> help(Predictor.params) #For fetching parameters
     >>> help(Predictor.predict) #For making predictions
     >>> help(Predictor.region_score) #For importance scores of AutoAtlas partitions
     >>> help(Predictor.score) #To evaluate prediction performance

-----------------------------------
Compute Losses for AutoAtlas Models
-----------------------------------
* For every epoch, compute the reconstruction error (RE) loss, neighborhood label similarity (NLS) loss, anti-devouring (AD) loss, and total loss on both the train and test sets by loading the saved model files.
* Either execute the script `aaloss` or use python to import the object `AutoAtlas` from the package `autoatlas.aatlas` and run its method `AutoAtlas.test`.
* Script `aaloss` is recommended for users without extensive knowledge of `python` and `pytorch`.
* For help on running `aaloss`, run the following command in a terminal::

     aaloss -h
* For help on using the method `AutoAtlas.test` from within python for calculating losses, run the following code in the python CLI::

     >>> from autoatlas.aatlas import AutoAtlas
     >>> help(AutoAtlas) #For help on creating AutoAtlas object
     >>> help(AutoAtlas.test) #For help on testing and computing losses

License
-------
AutoAtlas is distributed under the terms of the MIT license.

LLNL-CODE-802877

