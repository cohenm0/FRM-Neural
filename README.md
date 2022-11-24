# FRM-Neural
Project that uses neural networks to predict the interest rate for 30 year fixed rate mortgages

## Goal
The goal of this project is to compare the accuracy of a RNN against a time domain network through making 3 to 5 day predictions of the 30-year fixed rate mortgage.

## Approach
We will create an Elman RNN along with a temporal network. Each will be trained using about 80% - 90% of the same samples in our dataset. We will then test our networks using the remaining samples and ensuring everything stays consistent. We will review the data and conclude whether or not each network was accurate to predicting the 30-year rate mortgage and if so which was more efficient.  

TODO:
1. Add [30 year mortgage data](https://fred.stlouisfed.org/series/MORTGAGE30US) to the project 
2. Figure out how to organize the data (normalize? window size?...)
3. Port over MultiSunspot
4. Port over PredictSunspotElman
5. Get the code working
   1. Train the network
   2. Test the network
6. Generate results to a csv
7. Graph results in excel
8. Write a report

