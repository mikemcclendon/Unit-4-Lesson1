import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import statsmodels.tools.eval_measures as ste
import statsmodels.formula.api as smf
from sklearn import cross_validation as cv
import pandas as pd


loansData = pd.read_csv('loansData_clean.csv')
loansData['IR_TF'] = map(lambda x: 0 if x < .12 else 1, loansData['Interest.Rate.Clean'])
loansData['Intercept'] = 1
indVars = ['Intercept', 'FICO.Score', 'Amount.Funded.By.Investors']
	

def main():
	#Calculating via cross_val_score function
	kf = KFold(len(loansData['IR_TF']), 10)
	X1 = loansData[indVars]
	y1 = loansData['IR_TF']
	lr = LogisticRegression()
	msescores = cross_val_score(lr, X1, y1,scoring='mean_squared_error', cv=kf, n_jobs=1)
	r2score = cross_val_score(lr, X1, y1,scoring='r2', cv=kf, n_jobs=1)
	maescore = cross_val_score(lr, X1, y1,scoring='mean_absolute_error', cv=kf, n_jobs=1)
	
	
	#Below is an alternative means to calculating the cross validation stats. Seemed to give a more
	#intuitive answer, however not fully certain it is correct
	
	mselist = []
	maelist = []
	r2list = []
	r2listtrain = []
	r2listtest = []
	for train, test in kf:
 		X = loansData[indVars]
		y = loansData['IR_TF']
		#setting up train and test models
 		logit = sm.Logit(loansData['IR_TF'].ix[train], loansData[indVars].ix[train]).fit()
 		logittest = sm.Logit(loansData['IR_TF'].ix[test], loansData[indVars].ix[test]).fit()
 		testR = logittest.prsquared
 		trainR = logit.prsquared
 		#calculate MSE
 		mselist.append(ste.mse(logit.predict(X.ix[test]), y.ix[test], axis=0))
 		#calculate MAE
 		maelist.append(np.sum(logit.predict(X.ix[test])) - np.sum(y.ix[test]))
 		#calculate % difference in R2 of each train-test pair
 		r2list.append(abs((trainR - testR)/trainR))
 		r2listtrain.append(testR)
 		r2listtest.append(trainR)
 	
 	#printing results for both methods
 	print('First Method Using cross_val_score function:')
	print('MSE: '+str(np.mean(msescores))+' ,'+'RSquared: '+str(np.mean(r2score))+' ,'+
	'MAE: '+str(np.mean(maescore)))
 	print ('Second Method Using for loop (answers seemed more intuitive):')	
	print ('Mean of MSE is '+str(np.mean(mselist))	)	
	print ('MAE is '+str(np.mean(maelist))	)
	print ('The mean of R^2 percentage difference in each training and test samples is '+"{:.0%}".format(np.mean(r2list)))
	print ('The overall percentage difference of the total means of R^2 for training and test samples is '+"{:.0%}".format(
	abs((np.mean(r2listtrain) - np.mean(r2listtest))/np.mean(r2listtrain))))
	
if __name__ == "__main__":
    main()










