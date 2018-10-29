

![enter image description here](https://kaggle2.blob.core.windows.net/datasets-images/18/18/default-backgrounds/dataset-cover.jpg)
#  K-NN on Amazon Fine Food Reviews
## Objective 
 ###  Analyze ~500,000 food reviews from Amazon and determine the polarity on reviews

>**About this Dataset**
### Context
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories
### Contents
-   Reviews.csv: Pulled from the corresponding SQLite table named Reviews in database.sqlite  
    
-   database.sqlite: Contains the table 'Reviews'  
      
    

Data includes:  
- Reviews from Oct 1999 - Oct 2012  
- 568,454 reviews  
- 256,059 users  
- 74,258 products  
- 260 users with > 50 reviews

>### Table of Contents

 1. Loading the dataset , EDA 
 2. Pre-processing the dataset
 3. Text Featurization
    * Review Text -->Text Vector
 4. Applying different classification model with different hyperparameter
    * KNN
    * Naive Bayes
    * Logistic Regression
 5. Accuracy
 6. conclusion
>### Output Sample

*********k-fold knn (n_neighbors=k , weights='uniform') ***********
  ************* using k-fold to find best K-value ******************* 
    best accuracy is 88.97500048734379 on cv datatset using 10 fold at k-value 7
    genearalisation accuracy on best k-value at k = 7 is accuacy = 0.8746
			 

----
----
### 1. Loading the Dataset
1. downloading dataset from kaggle ['amazon fine food reviews'](https://www.kaggle.com/snap/amazon-fine-food-reviews) , and use 'panda' library to load the dataset
2. Exploratary Data Analyses of dataset
    * understanding the distribution of features

### 2. Text Preprocessing of dataset
* removing duplicate values
* removing stopwords
* cleaning unnecessary words
* remove punctuations and html tags
* stemming of words , etc

### 3. Text Featurization
Using text Featurization techniques to convert text into vector to make it ready to apply classification model
1. Bow
    * uni-gram
    * bi-gram
2. Tf-Idf
    * uni-gram
    * bi-gram
3. avg W2V
4. avg Tf-Idf W2v

### 4. Applying classification model

##### 1. KNN

##### 2. Naive Bayes

##### 3. Logistic Regression

##### 4. Support Vector Machine

##### 5. Random Forest


### 5. Accuracy



### 6. conclusion