# image_classification
 A python project  to classify the medical image using tensorflow ResNet50 model. The train and test datasets are available, each one has 10 classes. 

First run validate_images.py to find out any corrupted image files and then remove them from folder.
I do find out one image file which is corrupted.

Total 72101 files belonging to 10 classes in train folder. 
Using 61286 files for training.
Using 10815 files for validation.

Total 18030 files belonging to 10 classes in test folder.

All computation was done in Stony Brook University high performance linux cluster. 
Thw init.sh was executed first to create virtural environment, then create job.slurm file
to use GPU . All results are captured in png files and classification report. 
The prediction accuracy for training data is over 97% and it close to 95% for validation dataset.
The prediction accuracy for test data is around 95%.
