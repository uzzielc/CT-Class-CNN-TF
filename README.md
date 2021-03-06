# Hemorrhage Detection in Head CT Scans Using Convolutional Neural Network with head_ct_cnn.py

## Dependencies
This python script relies on:    
- [NumPy](https://numpy.org) (for some image manipulation and storing data for TensorFlow)  
- [Pandas](https://pandas.pydata.org) (for reading in the data labels)  
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html) (for image manipulation)
- [TensorFlow](https://www.tensorflow.org) (Neural Network Framework)

## Data
The data for this project was obtained from [Kaggle](https://www.kaggle.com).  
The link to the exact page for this data is [here](https://www.kaggle.com/felipekitamura/head-ct-hemorrhage).
  
  
The script requires that the files be formatted in the following:  
  
  
#### Main Folder:

/head-ct-hemorrhage/

#### Images:  

/head-ct-hemorrgage/head_ct/000.png
.  
.  
.  
/head-ct-hemorrgage/head_ct/199.png  

#### Labels:  

/head-ct-hemorrgage/labels.csv  

#### Script:  

/head-ct-hemorrgage/head_ct_cnn.py  


## Running the Script
Run the script in the command line by using:
python3 head_ct_cnn.py \<flag\>  
where \<flag\> can be:  
- -h (help)
- --te (train model & evaluate)
- --le (load weights & params & evaluate)
- --lte (load weights & params, train model, & evaluate)

## Performance

## Discussion
