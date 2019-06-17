# Grab-challenge
This is my solution for the tarffic management problem conducted by Grab as part of the AI for S.E.A challenge. More information on the challenge can be
found [here](https://www.aiforsea.com/traffic-management)

##Objective
The contestants are given the normalised demand for Grab taxis at certain geographical locations throught the day in 15 minute intervals. The geographical location is provided in the geohash6 format and are annonimized.The objective is to build a model that can predict the demand for T+1 to T+5 intervals given previous demand values.


##Training and running the models

*  python run_model.py  --i <path_to_Validation_Data.csv>    In order to run the model on validation data
*  python gen_stacked_data.py --t lstm   - create the data for stacked LSTM 
*  python train_lstm.py  - train the lstm model

Note: The run model expects the data in the same format as the training.csv file. If you need to generate data copy the training.csv file to data folder and run gen_stacked_data.py