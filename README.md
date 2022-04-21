# ML_Model_Comaprisons
I used several different machine learning models (classification + prediction) to determine which model was superior.

This was my midterm project for a machine learning course I took in my masters program in data analytics at Georgia State University. 

Below is my report/analysis on my project findings.



Business Question I: Can I predict whether the perpetrator of a mass shooting had displayed previous signs of mental illness?

Business Question II: Can I predict how many weapons the perpetrator had on them when they committed the mass murder?

Dataset Overview

	The dataset contains all of the mass shootings that have occurred in the United States. The first mass shooting recorded in the United States was in 1982. The dataset only has records dating to 2015, so there are a total of 33 years of data. There are a total of 71 entries in the dataset that span that time span. The Investigative Assistance for Violence Crimes Act of 2012 defined a mass killing as an event committed by an individual where 3 or more victims are murdered, excluding the perpetrator. The dataset includes 25 features to help describe each event. The vast majority of events in the dataset have values, so there is a high level of integrity within the dataset as a whole. Here is initial look at our dataset:



 

Some of the variables were not conducive to being used in a machine learning model. For this reason, these columns were dropped. For example, the 'CASE' and the 'SUMMARY' columns were dropped. I tried to brainstorm interesting ways to engineer the features, but it turned out to be unproductive. 



Data Exploration
I used several  methods to explore the data. The first method I used was the .describe() method. This method is only useful with numerical columns. Due to my dataset have multiple numerical features, I used this for exploring these features:

 

Other than the obvious columns that needed to be dropped, this was a productive way of getting a better understanding of the numerical columns. For the categorical columns, one of the methods I used was the value_counts() method:


Above, you can see the total value counts for the 'STATE' feature. This helps give the model designer whether or not the feature will be helpful or not for training the model.
	The next step was to convert the categorical variables into dummy variables. To do this, I used the one hot encoding method. When doing this, I separated the data frame into 2 new data frames. One labeled 'CATEGORICAL' and the other 'NUMERICAL'. After this, I only ran the one hot encoding method on the 'CATEGORICAL' column. I then had a dataframe that looked similar to this:


This is a subsection of the complete 'CATEGORICAL' dataframe. After this, I concatenated the 'CATEGORICAL' and 'NUMERICAL' dataframe. This resulted in a dataframe that had the categorical variables one hot encoded along with the numerical variables in an unaltered format. Other than applying feature scaling techniques for certain models, this would be my final data frame that I would use for training both my predictive and categorical models.
	
Data Preprocessing
	I used feature scaling on the majority of my models. I used both the standardization and normalization techniques. Normalization is suggested when the data does not fit a Gaussian distribution. This was the case for my data, though, I used both techniques in my code and was able to analyze the differences in the results. In some models, feature scaling worsened the accuracy score, for instance with linear regression, though it helped with distance based models like K Nearest Neighbors and Support Vector Machines. 

Classification Models
	For the classification models, I used logistic regression, K-Nearest-Neighbors, Support Vector Machine, and a Classification Tree. The purpose of these models was to determine whether they could accurately classify whether the perpetrator had displayed previous signs of mental illness. The values in the 'PRIORSIGNSOFMENTALILLNESS' were either Yes or No. This therefore worked as a suitable target variable for the situation.
	The classification tree model used 3 different models: smallClassTree, fullClassTree, and bestClassTree. All of the trees had to be individually trained. There are significant differences in the number of branches with each tree model. The bestClassTree had the best accuracy score coming in at .556. The fullClassTree and smallClassTree both had equal accuracy scores of .5 The bestClassTree had a 0 score for all of the confusion matrix metrics because 100% of the accurate classifications were true negative. Therefore, many of the metrics that used the true positive value in the formula resulted in the value being a 0 due to it not being able to be multiplied in the denominator or numerator or exist by itself in the numerator when the formula was a quotient. 
	The logistic regression model is the next classification model. The accuracy metrics came in quite high in contrast to the rest of the classification models. The accuracy score was .50 and the Area Under the Curve came in at .593. The AUC scores came in lower when the data was either normalized or standardized. The models run with those data both came in with an accuracy score of .5 and an AUC of .494. This was odd considering that this model was a gradient descent model. Having only 71 entries was likely the cause of this.
	The next model used was KNN. I used 'NUMWEAPONS' and 'LOCATION' as the two independent variables in this model. I used a function to provide the optimal k values. 9 was the optimal k value and was used. The AUC for this model was 0.7. The accuracy metrics were rather suitable. For ‘PRIORSIGNSOFMENTALILLNESS’ being ‘Yes’, the precision was 0.74, recall 0.85, and f1 .079. The scores were lower when the features were either scaled or normalized.
	The Support Vector Machine had a close enough AUC score for me to give it a tie for 1st place. The # of support vectors were 47 out of the 71 original vectors. I set my C value to 10.0, kernel = "linear", and random_state = 1. The accuracy metrics were fair. The model had a better AUC score when the features had not been scaled. 0.44 when the data was standardized and 9.593 without the data being feature scaled.
	Continuous Models
	I chose to use linear regression, regression trees, and support vector regression. My business question for this was: Can machine learning models help predict how many weapons the perpetrator had on them when they committed the crime? My target variable is 'NUMWEAPONS', and it is in integer format.
	For linear regression, I tested multiple, simple, ridge, and lasso regression. The accuracy scores for the simple and lasso regression were not good, while they ridge and multiple regression scores were acceptable. The multiple linear regression model proved to tie for 1st along with the SVR for the most accurate models. The multiple linear regression model came in with a score of 100% accuracy. The score was higher when the features were not scaled, despite it being a gradient descent model. The ridge regression was an acceptable accuracy score that came in with a 0.83.
The regression trees were acceptable in their accuracy metrics. Their MAE came in with a 0.09, MSE of 0.18, RMSE of 0.426 and RMSLE -0.85. Overall, I was content with these scores. They weren't as successful as the multiple linear regression, though, because they didn't have 100% accuracy, I thought it might represent a higher quality model if there were more entries in the dataset. Though, the multiple linear regression model may still be more accurate. 
The support vector regressor had a range of accurate outcomes depending upon what the kernel was set to. There were 3 different svr variations that were used: SVR_best, SVR_linear, and SVR_rbf. The variation for the 2nd and 3rd models depends on what the kernel is set to. The kernel that was set to linear was significantly more accurate than the other 2 scores. The SVR_linear score was .992. This was high enough for me to consider it a tie with the multiple linear regression model. 
Conclusion
Each business question had 2 models that tied for 1st place. For the continuous output regression problem, linear regression and the support vector regressor models were dominant. To be specific, it was the multiple linear regression and SVR_linear that were superior to the other models within their parent categories. While the multiple linear regression had a perfect 1.0 accuracy score, the SVR_linear model had a .992 accuracy. Although technically these are uneven, I consider it close enough to be a tie.
For the classification business question, the logistic regression and SVM model had the highest AUC scores. Both of the models had the exact same score of .593. This was higher than all of the other classification models. It should not be disregarded that both the SVM and SVR models had to have their kernel set to 'linear'. Although there is a distinct difference between linear and logistic regression, possibly this is the relationship between the SVR linear model having the same score as the logistic regression model. It would provide a lot of insight if I were able to have more entries in the dataset so that I could make inferences on which models were better.



Here is an initial look at our dataset before data preprocessing. You can see there are some missing values that are represented by the NaN value (Not a Number). You can see all of the 9 attributes that have been described above. I used a python function to identify all of the different values for our categorical variables. I reached this observation: Housing ('own', 'free', 'rent'), Saving accounts ('Nan', 'little', 'quite rich', 'rich', 'moderate'), Checking account ('NaN', 'little', 'moderate', 'rich'), Purpose ('radio/TV', 'education', furniture/equipment', 'car', 'domestic appliances', 'repairs', 'vacation/others'), and Sex ('male', 'female').



Data Exploration
	Below is a statistical exploration into our dataset's numerical attributes. I can observe a variety of basic statistical values for the specific columns. Using general rule-of-thumb practices, I am able to look for any red flags in our data. One red flag would be the min value for credit amount. The value is significantly lower than the 25% threshold value. This would encourage investigation into that value for potential elimination of the value if it is a significant outlier.


Here is a numerical table representation of our missing values and variable data types. I want to avoid having any missing values, so ideally all of the values in the Non-Null Count would be 1000. There are 183 missing values in the Savings account column and 394 missing values in the Checking account column.



	Data PreProcessing
Linear regression cannot work with categorical variables that are written as text because they are categorized by Python as the object data type. They need to be in integer format, so I used a Python method to one-hot-encode the categorical variables. This way, the linear regression model can account for the 
categorical variables when it is being trained on the dataset. Below is an example of how implementing this technique on our dataset affected the checking_account column. You can see the before and after effect this technique had on our values. Each categorical variable has its own column and is scored with a 1 for each instance that it appears in the column.

         
            Before							    After



I renamed the column names because several of the column names used spaces; I rewrote each column name with an underscore to replace the space. I did this because spaces are not conducive to the Python syntax. Below, you can see that I now have all 1000 values for our Non-Null Count. This means that I have no missing values in our dataset because all of our NaN values are now represented by integers. Our risk column still has an object datatype. This is because the linear regression model can accept categorical binary variables. The Risk column is the only column from the original dataset that had this property. 


Feature Exploration
The bin graphs (graph set 1) on the next page are examples of graphs that I constructed from our original data. The first graph on the bottom left represents the number of customers with credit lines of varying amounts. This graph aids in representing the most common loan amounts to be given out by the bank. There is a clear trend that most customers are given shorter amounts of time to repay the loan. This trend can be described as exponentially decreasing. The bin graph on the right shows the most common credit repayment durations. There is not an obvious trend with this graph. It does not seem like there is any recognizable pattern with this data through visual analysis. 
(graph set 1)
	Our second set of graphs is graph set 2. The bar graph on the bottom left represents the ratio of good and bad risk assessments for each job skill level/residency. This graph clearly shows that type 2 is the most common, a skilled worker. Skilled workers and unskilled workers that are residents have the best credit risk assessments. There is not much data for type 0. Type 3 lies somewhere between type 0 and type 1. 
The bar graph on the bottom right represents the good/bad risk ratio for the Housing attribute. It is clear that type 2, free housing, represents the lowest credit risk and is the most common of the customers. The renting and owning customers have about an equal ratio of risk and are far less common in the customer base.


(graph set 2)
	The next set of graphs I am analyzing is graph set 3. The bar graph on the bottom right is a graph that represents the risk ratio for males and females. Males have a significantly better risk rating than females according to this data. Males have a more favorable ratio and they have more data to support their trend. It should be acknowledged that both genders have ratios that have favorable risk ratings. The box plot on the bottom left represents the age distribution for good and bad risk assessments. Both good and bad assessments have rather similar age distributions. The good risk assessments are slightly older in age than the bad ratings, but not significantly. This is true for the 25% mark, the median and the 75% mark for both. Both plots seem to have a similar amount of outliers above their respectives maximums. The maximum value for the bad rating seems to be significantly lower than the maximum good value, representing the age. 

(graph set 3)


Linear Regression
	I split the dataset into a training and a test dataset. After this, I calculated our MAE, MSE, RMSE, RMSLE and R^2 scores. They were recorded as follows: MAE 0.19, MSE 0.062, RMSE 0.25, RMSLE -1.38, and R^2 .37. The R^2 tells us that I only have 37% of our y variation explained by our x variation. Having our R^2 at 37% is telling us that our model is only explaining 37% of our target variable. In other words, our x variables or features, are only explaining 37% of our credit loan amount. This value is low for our liking, and I would like to have an r^2 of  >50% before having confidence in our model.
Backward Elimination :
Backward elimination is a feature selection technique used while building a machine learning model. It is used to remove features that do not have a significant effect on the dependent variable or prediction output. It begins with a model containing all the independent variables of interest. Then at each step the variable with the smallest F-statistic is deleted until left with only the significant features required for the prediction of the target. In the case of our dataset, no feature is removed. This denotes that all the features have a significant influence over the target prediction.
 
Forward Selection :
Forward stepwise selection (or forward selection) is a variable selection method that begins with an empty model that contains no variables, called the Null Model. Then the process starts adding the most significant variables one after the other. It adds only the significant set of variables and omits the other features. This provides the best set of variables for the model to predict output. In the case of our dataset, and as occurred with our backward elimination, no feature is not added which denotes that all the features have a significant influence over the target prediction.
 
Lasso Linear Regression
	I ran both the Lasso and LassoCV regression. I tried both of the models in order to find the best alpha. The alpha value that I found to be optimal was .00021. The metrics for our Lasso regression were: ME -0.0410, RMSE 0.3201, MAE .2595, MPE -2.1434, MAPE 7.9372 and R^2 of 0.531.

Conclusion
	I decided that linear regression was superior to lasso regression. Lasso had a superior r^2 value of 0.531 and linear regression had a smaller r^2 value of 0.37, The linear regression had a smaller MAE of 0.19 and RMSE 0.25 while LASSO had larger values of MAE .2595 and RMSE 0.3201. I favored the smaller MAE and RMSE and went with that as our deciding factors.
![image](https://user-images.githubusercontent.com/62865950/164372144-12787bef-cb9b-4bf6-8c11-229d6c079976.png)
