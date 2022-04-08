Steps to train a ML model:

1. Explore the dataset and do some prelimiary EDA on it
2. Write the dataset class for it (includes __getitem__() function to be carefully implemented)
3. Decide and import the kinds of transforms and augmentations we need to do on the dataset
4. Know what kind of optimizer and loss functions you are going to use based on the data
5. Figure out which model works best. 
6. If you want to write a custom model, make sure you have the input and output dimensions sorted. 
7. If you are using a pretrained model, make sure the model is properly imported. 
8. Make sure the model and data are uploaded to cuda if cuda is available. 


Basic EDA to perform:
1. See what kind of data it is (images, text, multimodal)
2. See how labels are given (csv file, xml file, overlapping images)
3. Know how you want to format the dataloader. 

Use of Optimizer: 
1. 