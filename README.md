The is a project related to image classification.

Train data path: data/All_61326/train_61326
Test data path: data/All_61326/test_61326

Model built is saved in the folder: model/

Source code: src/

Following file descriptions: 
1. src/dataset.py: contains a class as a container for image data
2. src/tensorflow_model.py: contains the code for CNN to build image classifier
3. src/train.py: contains code for training and testing. 

    Training code is commented from line 46-56. In case retraining is required, please uncomment 
    these lines of code and call the train.py class.
    
    Currently train.py class has test module as well. This module picks up the saved model and 
    runs classifier on test images.
    
Google doc link for details:
https://docs.google.com/document/d/1yUz2zF5UIXyinln17vguvRkil3pAWX9Q1ntuDcwM3bo/edit?usp=sharing