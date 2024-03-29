# Brain Tumor Detection Using Pytorch Model
This is a brain tumor classification model created using transfer learning with resnet-50 model to classify the brain tumor MRI images into 17 classes. The model has achieved an accuracy of 97% on test data.


## Steps to Run the Website

* Download the github folder to you local server using following command:

  * git clone https://github.com/anu81785/brain_tumor_classification_into_17_classes.git

* Change the directory to downloaded folder:
   
   * cd brain_tumor_classification_into_17_classes

* Create a virtual environment using python:
   
   * python -m venv env_name

* Activate the virtual environment:
   
   * source /path/to/venv/bin/activate

* Now download all the required packages inside this virtual environment using following command:
    
   * pip install -r requirements.txt

* Start the following commands to apply the migrations and runserver
    
   * python manage.py migrate
   
   * python manage.py runserver

   ![Screenshot from 2023-09-01 17-20-16](https://github.com/anu81785/brain_tumor_classification_into_17_classes/assets/89373629/3de9c5eb-f486-4af3-b1be-61d97c5df148)

* Open the url http://127.0.0.1:8000 in your browser

  ![tuxpi com 1693572025](https://github.com/anu81785/brain_tumor_classification_into_17_classes/assets/89373629/a0f94517-e2a9-42a6-ac50-6d33ac130200)

* Upload the brain MRI image for which you want to detect the tumor type

  ![tuxpi com 1693572451](https://github.com/anu81785/brain_tumor_classification_into_17_classes/assets/89373629/1716e35b-7f0f-42e0-9b22-7874fbae8d50)

## Link to Dataset (MRI images of brain) from kaggle

* https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes
  

## Plots for Loss and Accuracy for Training and Validation Set

![plots](https://github.com/anu81785/brain_tumor_classification_into_17_classes/assets/89373629/a1ecfda8-cf3a-4b8b-b2ae-49baabe03ef5)

  

## Confusion Matrix for Test Data


  ![index](https://github.com/anu81785/brain_tumor_classification_into_17_classes/assets/89373629/a7b5d9cc-bd28-4918-a00b-a3a97d712ff4)



