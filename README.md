# Classification - Breast Cancer Wisconsin Diagnostic

Supervised machine learning project to predict cancer diagnosis using the popular Breast Cancer Wisconsin dataset

Contents
* [Introduction](#introduction)
* [Literature Review](#literature-review)
* [Methodology](#methodology)
* [Subset Selection](#subset-selection)
* [Logistic Regression](#logistic-regression)
* [Linear Discriminant Analysis](#linear-discriminant-analysis)
* [Ensemble](#ensemble)
* [Support Vector Machines (SVM)](#support-vector-machines-SVM)
* [Neural Network](#neural-network)
* [Data Exploration & Descriptive Statistics](#data-exploration--descriptive-statistics)
* [Boxplot](#boxplot)
* [Outliers](#outliers)
* [Pairwise Scatter Plot](#pairwise-scatter-plot)
* [Clean Data](#clean-data)
* [Findings](#findings)
* [Subset Selection](#subset-selection)
* [Logistic Regression](#logistic-regression)
* [Linear Discriminant Analysis](#linear-discriminant-analysis)
* [Neural Network](#neural-network)
* [Conclusion](#conclusion)
* [References](#references)
* [Appendix 1 : R Code](#appendix-1--r-code)

Packages
* pacman::p_load * data.table * fixest * BatchGetSymbols * finreportr * ggplot2 * lubridate * readxl * dplyr * tidyverse * extrafont * ggthemes * RColorBrewer * scales

## Introduction
This analysis examines the predictive accuracy of five advanced analytics and machine learning (ML) algorithms on the publicly available and popular ‘Breast Cancer Wisconsin (Diagnostic)’ dataset. Models will be developed using Logistic Regression, Discriminant Analysis, Ensemble, Support Vector Machines (SVM) and Neural Network. The dataset developed consists of features from digitized images taken from real life tests on breast mass using Fine Needle Aspiration (FNA) and are descriptive characteristics of cell nuclei existing in the image. The dataset partitioned into Train/Test dataset with 80/20% partitioning. Results show high accuracy with Neural Network 97% and linear discriminant analysis 96.5% but the other models fell short of expectations for the prediction with logistic regression results at 61%.

## Literature Review
According to the World Health organisation (WHO) almost 10 million deaths in 2020 were as a result of cancer, a statistic of 1 in 6 deaths. Most common occurrence of cancer variant in new cases of the world’s population in 2020 was breast cancer with 2.26 million cases, closely followed by lung cancer then colon and rectum to make up the top 3 most common cases (WHO, 2022). In 2015 breast cancer was the most common cancer to kill women globally.
In the UK alone there were 55,920 cases of breast cancer of which 11,547 deaths between 2016 and 2018 an astonishing 20.7% (Cancer Research, 2022), notably the numbers are extremely higher in females than males.
The burden that breast cancer has on medical treatment and finances globally is catastrophic with an estimated cost of US$1.16 trillion in 2010, given that 30-50% could be avoided through lifestyle decisions and greater public health measures.
Diagnosis of breast cancer is becoming increasingly precise and can be carried out with the help of advanced analytics and machine learning techniques. Historically the responsibility solely lay with the expertise of the physician for an accurate diagnosis of malignant breast cancer over benign lumps.
Supervised learning is when the values of the variables are known as opposed to unsupervised learning where these values are unknown or unavailable. Classification is identifying an outcome into a single category of two or more categories based on features of each object. Explanatory variables in supervised learning are often referred to as features and training set is often used to refer to the data.
Furey et al. (2000) outline how SVM have shown to have performed well in different areas of biological analysis and have shown an ability to correctly separate entities correctly into classes and identify instances with no established classification in data. They illustrate how SVM can classify new samples whilst also having the ability to identify misclassification by experts and have demonstrated this on ovarian cancer tissue samples.
Kourou et al. (2015) illustrate how the latest advancements in machine learning (ML) are being utilised with the aim to model the risk of cancer and outcomes in patients showing symptoms. They outline that one of the most challenging tasks for a physician is disease accurate prediction with cancer susceptibility and recurrence data, artificial neural networks ability to predict mammographic observations to 96.5% and in clinical, pathological and SNP observations support vector machines obtained accuracies of between 68% and 95%. They identify research where using a dataset of 162,500 observations based on the SEER cancer database for support vector machines (SVM), artificial neural network (ANN) and self-supervised learning (SSL) calculated accuracies were 51%, 65% and 71% respectively. However they identify drawbacks with experimental design, data sample collection and classification validation and the claims of adequate and effective techniques have resulted in limited clinical practice penetration. Other limitations in the study noted were the small data samples to be reviewed where size matters due to the requirement to effectively model on an efficiently large training dataset.
Park et al. (2013) in their literature highlight various studies of prominent machine learning (ML) models for breast cancer survivability prognosis, they look at the popular methods SVM, ANN and SSL while using surveillance epidemiology and end results (SEER) breast cancer database. They outline that the ANN model in terms of stability and accuracy did not perform adequately while SVM, when carefully tuned performed reasonably well but with fluctuating performance. However as outlined in Kourou et al. (2015) they highlight the performance of SSL in this instance.
Rosado et al. (2013) aim from their study to find; based on use of a SVM classifier algorithm,  an intelligent, efficient model for accurate prognosis prediction in oral squamous cell carcinoma (OSCC) patients while evaluating the model by discriminant analysis. OSCC a life threatening disease with survival rates of between 50% and 60% over 5 years prior to 2013. Outlining how SVM has the flexibility and ability to provide high accuracy while dealing with high dimensional data. They aim to find patterns in the data for patients’ survivability while also producing a highly predictive model with relevant variables identified. Data was gathered from 69 OSCC suffering patients, selected randomly. They highlight how rates of classification were 97.56% and 100% for living and deceased patients with a 98.55% overall classification rate illustrates the ability of SVM to improve the error caused by excessive quantities of variables.
Lu et al. (2020) investigate in their study the ability to accurately predict ovarian cancer (OC), a major cause of death in women with 184,799 recorded deaths in 2018. A chemo sensitive cancer that show initial efficacy, unfortunately recurrence rates rose from 60% to 80% with greater recurrence percentages recorded for higher stages. Currently clinical testing for diagnosing OC is by a small quantity of biomarkers in gene/protein criteria or epidemiological evidence, though ML advances and ability to handle high dimensional data can prove an accurate model for prediction. Their study focuses on data from 349 patients with 49 measures with result accuracies of 87.2%, 84.7% and 79.6% for Decision Tree, Logistic Regression and Risk Ovarian Malignancy Algorithm (ROMA) respectively.
 
##	Methodology
##	Subset Selection
One of the most critical operations of pre-processing of the data is subset selection which identifies the attributes or features that make the highest meaningful contribution to the machine learning activity. The role of subset selection is used when hundreds or even thousands of attributes or variables exist in the dataset which is referred to as high dimensional data which can be a challenge for machine learning algorithms to deal with and handle the data. With high dimensional data a high quantity of computational power and greater amounts of time required, along with this a model with a lot of features can be very difficult to understand.
Given the issues outlined regarding high dimensional data it is therefore necessary to subset features within the dataset instead of working with the full set. Reasons to reduce the dimensionality of the data can include:
1.	Faster and less costly
2.	Greater understanding of underlying model
3.	Improved learning model efficacy
There are various methods of sub-setting data in order to reduce dimensionality, they include
a.	Best subset selection – this selection method is based on comparing all models containing one predictor, all models containing two predictors etc and the best model selected with corresponding subset size
b.	Forward and backward stepwise selection – with large quantities of predictors the best subset method becomes unfeasible and statistically risky, it can take a large amount of time to compute and overfitting can occur. When this is the case forward and backward stepwise selection can be used.
This study will use best subset selection to obtain best performing features from the data which will then be used to develop accurate models using various machine learning techniques.

##	Logistic Regression
This is a classification modelling technique that predicts the probability that a binary response Y belongs to a certain category instead of directly modelling the response Y. In below Figure, the left hand panel illustrates the result of direct modelling of Y as the result similar to the result if linear regression with results outside of [0, 1] interval. However when Y is predicted to belong to a particular category the results are of that Logistic Regression prediction can be seen in below Figure right hand panel. Thresholds can be set for each predicted result for class selection.
Logistic regression can be categorised into different types:
1.	Binary Logistic Regression – a categorical 2 possible outcome response
2.	Multinomial Logistic Regression – unordered categories of 3 or more quantity
3.	Ordinal Logistic Regression – ordered categories of 3 or more quantity
This study will look at binary logistic regression in order to classify the FNA lump as Malignant or Benign. In reality what the classification is doing is predicting whether the data belongs to a particular category, in this case Malignant or Not Malignant (ie. Benign)

 
Figure: Modelling of dependent variable using Linear (left) and Logistic (right) regression

##	Linear Discriminant Analysis
Similar to Logistic Regression when distribution is assumed normal though and more accurate when sample sizes are small. It is a dimensionality reduction technique and as LR suffers from instability when substantial separation exists between both classes, Discriminant Analysis does not suffer from the same issues. It can be used to reduce the feature count in pre-processing stages of data analysis and reduces computing cost significantly.
Discriminant functions can be roughly categorised into three classes:
1.	Distance-based
2.	Bayes
3.	Probability-based
Pioneered by Fisher in 1930’s distance-based were the earliest and often referred to as Fisher Linear Discriminant Analysis (LDA). Bayesian has a decision-theoretic foundation and came later. And thirdly, probability-based arguably most standard approach and estimates the Bayes rule. Bayes rule being optimal a decent classifier should interpret a satisfactory approximation.
It is used in medical applications to classify a patient’s disease status based on various parameters and treatments being undertaken by the patient. It can assist medical experts with a more accurate course of treatment. It will be used here to try to accurately identify the disease status of cancer patients based on features compute from finite needle aspiration (FNA) data.



 
##	Ensemble
The ensemble idea is combining a collection of simpler model strengths. Various classification ensemble methods include Bagging, Random Forests and Boosting. 

##	Support Vector Machines (SVM)
Support Vector Machine (SVM) first developed by Vapnik and a support vector classifier extension, enlarges the boundary/feature space using kernels to enlarge to maximum separation between closest points of two classes.
SVM exhibit, as well as nice theoretical properties, exceptionally superior performances on classification tasks and also avoid what is often seen, the curse of 
The dimension of this feature space can get very large and in some instances infinite which would seem the data would be separable leading to overfitting. However SVM deals with this issue and solves function fitting problems and forms part of a greater class of problems (Hastie et al., 2009 & Clarke et al., 2009 & James et al., 2014).
Key properties for a separating hyperplane is that it is furthest from the data meaning the boundaries between two data classes are as far apart as possible ensuring that small changes in the data reduces the likelihood of changing the value of the classifier. 
 		 

##	Neural Network
The heart of Deep Learning, Neural Networks first came about in the 1940s when proposed by Warren McCullough and Walter Pitts (University of Chicago researchers) Hardesty (2017). Initially they brought major attention from computer science and neuroscience before losing popularity. Now with increased computational processing power have seen a major resurgence.
The name comes from the mimicking of the human brains biological neurons which sends signals to each other. They’re compromised of layers, input layer, hidden layers and output layer (see Figure below). The nodes are activated if the output from the previous node has met a specified threshold value and the data is then sent to the next network layer. Otherwise the layer stops and data does not get passed on. 

 
Figure: Artificial Neural Network Architecture (IBM 2020)

They have a reliance on training data in order to learn and over time improve their accuracy, once fine-tuned they become powerful tools in artificial intelligence and machine learning enabling with high velocity the classifying and clustering of data.
 
##	Data Exploration & Descriptive Statistics
The dataset supplied via .csv file is concerned with generated features computed from images of fine needle aspiration (FNA). The dataset consists of 569 observations and 32 variables, within this dataset the diagnosis feature recorded 357 benign and 212 malignant cancer observations. All features were recorded with 4 significant digits and there are no missing values within the data. When imported into R Studio the dataset consists of 568 observations and 33 variables. The below table outlines the variables and their respective classes.
 
Table: Dataset Variable Classification
Summarising the raw imported data provides a rudimental overview and can be used to make better decision around features to be used throughout the analytics process. Below figure shows the overall imported data summary.
 
Figure: From the summary of the data issue
 
To gain an overview of associations between variables and dependent outcome variable Table was developed using the Pearsons Chi-Squared Test and Table developed using T-Test to review characteristic variables.

 		 
Table XX: Chi Sq Test					Table XX: T-Test

##	Boxplot
As an extremely effective way to visualise the data boxplots have the ability to identify features with the data with outliers. This issue can then be invetsgated further and the datapoints can be removed or retained if deemed necessary. The below Figure XX visualises the variables; radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave.points_mean, symmetry_mean, fractal_dimension_mean from the dataset.
 
##	Outliers
The table below highlights the outliers existing the ‘_mean’ variables within the dataset, individually there are not high quantities though these should be converted to NA and observations kept. This will reduce their impact on the model.
 

##	Pairwise Scatter Plot
The figures below illustrate correlations in pairwise fashion between the features radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave.points_mean, symmetry_mean, fractal_dimension_mean from the dataset. It is very visible from the plots that some data are highly correlated which is a cause for concern due to multicollinearity in the data.
  
 
Figure: Pairwise plots of ‘_mean’ features within the dataset

##	Clean Data
The imported dataset included the  variable ‘id’ and R Studio added another  logical variable ‘….33’ which need to be removed as they serve no purpose in the modelling and could have a negative effect on the outcomes.
The below Figure visualises the variables as a result of cleaning the data by removing outliers; radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave.points_mean, symmetry_mean, fractal_dimension_mean
 
Figure: Boxplot of cleaned ‘_mean’ features
Removing all NA values in from the ‘_mean’ variables would result in a loss of 11% of the overall dataset. It was decided to keep these ‘NA’ values in the dataset and ignore values.
 
Table: Total Outlier observations in dataset 
Figure below show the summary of the cleaned data highlights the inclusion of NA values within the dataset.
 
Figure: Summary of dataset with removed outliers
 
By using the regsubset() function the best variables models can be determined and as shown below, each top row contains a square of black for each selected variable according to the associated statistics for the optimal model. It is possioble to determine the model with the lowest BIC.
   
##	 Findings 
##	Subset Selection
Through subset selection and running the leap() function on the data the models with the minimum Cp in the data include 11 features and they are;  compactness_mean, concavity_mean, radius_se, smoothness_se, concavity_se, radius_worst, texture_worst, area_worst, concave points_worst, symmetry_worst, fractal_dimension_worst.
 
The plotting of subset selection RSS, adjusted R2 , Cp and BIC illustrates a line model constructed from all the generated models with connected points. The red point on the adjusted RSq plot highlights the maximum value whereas the points on the Cp and BIC plots illustrate the minimum. This is computed through the code and the best model is shown to have 11 variables confirmed through the coded minimum BIC and the maximum adjusted R2.

 

 
##	Logistic Regression
From the LR training model results shown below the confusion matrix demonstrates fair level of accuracy although observations of 4 (1%) & 7 (1.5%) of the data across the entire training set. This might seem very small but taking scaling into account for other datasets the numbers can rise significantly. These values are reflected in the prediction accuracy of 65% on training data which results in a 34.3% training error rate. 
 	 
Figure: Logistic Regression (train set) confusion matrix (left) & prediction accuracy (right)
Following on from the trained model on the training dataset this model should be tested for accuracy on the test set. The results for a robust model would mean similar results to the training model, usually test results are a little lower and this is highlighted in through these findings.
 	 
Figure: Logistic Regression (test set) confusion matrix (left) & prediction accuracy (right)

##	Linear Discriminant Analysis
From the plot of the lda it is possible to identify that the observations are not centred on zero meaning that there is significant differences in the data although similar in possibility that the cancer variable is that of the group (0 , 1).
 
Figure: Linear Discriminant Analysis Plot

Using all features of the dataset to make predictions on the diagnosis of the patient, the below Figure XX highlight how 96.3% train and 96.5% test accuracy was achieved with the model.
 		 

 		 

However by using only the outlined 11 features in the subset selection the model showed exactly the same accuracies although the false readings for 1 (Malignant) were reduced by 1 and false reading for 0 (Benign) increased by 1.
 		 

 		 




 
##	Neural Network
Training a Neural Network model on the features radius_mean, smoothness_mean, perimeter_se with 1 hidden layer and a threshold value of 0.01 the below results were obtained when training set (left ) and test set (right). Results show 89.2% accuracy for the training dataset and 93.8% accuracy on the training set which are high results.
 		 

 		 

When the features outlined the best subset selection; compactness_mean, concavity_mean, radius_se, smoothness_se, concavity_se, radius_worst, texture_worst,  area_worst, symmetry_worst, fractal_dimension_worst, with 1 hidden layer and a threshold value of 0.01 the below results were obtained when training set (left ) and test set (right). Results show 98% accuracy for the training dataset and 97.3% accuracy for the test set. Below also can be seen the results matrix for this neural network model.
 		 

 		 
 
##	Conclusion
This study presents a variety of machine learning methods popularly used to develop predictive models and aspired to develop similar models with high predictive accuracy. The topic has been broadly researched and the prediction of patient cancer disease is a very interesting topic of research. While all models within the study don’t perform similarly to those in literature given the time constraints the insights gained are valuable and in a real world application where time may be less scarce the models could be developed further with addition of further features to fine tune the models.
The notable best performing model was the Neural Network model with 97% accuracy prediction on the test dataset which is a confident accuracy based on this outcome. In second place on the performance was the Linear Discriminant Analysis with 96.5% accuracy on the test dataset and lagging considerably behind is the Logistic Regression with only 61%.
The results in this study are high and confidence should be taken from this model training. In future to further the analysis an ensemble model should be developed, this had been in the scope for this analysis but with time constraints it needed to be removed.
The development of modelling medical predictions like this can alleviate a major burden on global health services and further study should be undertaken. In future, a significantly bigger dataset should be taken into consideration and models developed with greater confidence.


 
##	References
Cancer Research UK (2022). Breast cancer incidence (invasive) statistics [website]. Accessed by: https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/breast-cancer/incidence-invasive#ref- (Accessed May 2022)
Clarke, B. S., Fokoué, E. and Zhang, H. H. (2009) Principles and theory for data mining and machine learning. Springer (Springer series in statistics). Available at: https://search.ebscohost.com/login.aspx?direct=true&db=cat02616a&AN=qub.b16601725&site=eds-live&scope=site (Accessed: 15 May 2022).
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
Furey, T.S., Cristianini, N., Duffy, N., Bednarski, D.W., Schummer, M. and Haussler, D., 2000. Support vector machine classification and validation of cancer tissue samples using microarray expression data. Bioinformatics, 16(10), pp.906-914.
Hardesty, L (2017). Explained: Neural Networks [website]. Accessed by: https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414 (Accessed May 2022)
IBM (2020). Neural Networks [website]. Accessed by: https://www.ibm.com/cloud/learn/neural-networks (Accessed May 2022)
James, G. et al. (2014) An introduction to statistical learning : with applications in R. Springer (Springer texts in statistics). Available at: https://search.ebscohost.com/login.aspx?direct=true&db=cat02616a&AN=qub.b19003845&site=eds-live&scope=site (Accessed: 15 May 2022).
Kourou, K., Exarchos, T.P., Exarchos, K.P., Karamouzis, M.V. and Fotiadis, D.I., 2015. Machine learning applications in cancer prognosis and prediction. Computational and structural biotechnology journal, 13, pp.8-17.
Lu, M., Fan, Z., Xu, B., Chen, L., Zheng, X., Li, J., Znati, T., Mi, Q. and Jiang, J., 2020. Using machine learning to predict ovarian cancer. International Journal of Medical Informatics, 141, p.104195.
Park, K., Ali, A., Kim, D., An, Y., Kim, M. and Shin, H., 2013. Robust predictive model for evaluating breast cancer survivability. Engineering Applications of Artificial Intelligence, 26(9), pp.2194-2205.
Rosado, P., Lequerica-Fernández, P., Villallaín, L., Peña, I., Sanchez-Lasheras, F. and De Vicente, J.C., 2013. Survival model in oral squamous cell carcinoma based on clinicopathological parameters, molecular markers and support vector machines. Expert systems with applications, 40(12), pp.4770-4776.
Trevor Hastie, Robert Tibshirani and Jerome Friedman (2009) The Elements of Statistical Learning : Data Mining, Inference, and Prediction, Second Edition. New York: Springer (Springer Series in Statistics). Available at: https://search.ebscohost.com/login.aspx?direct=true&db=nlebk&AN=277008&site=eds-live&scope=site (Accessed: 15 May 2022).
WHO (2022). Cancer [website]. Accessed by: https://www.who.int/health-topics/cancer#tab=tab_1 (Accessed May 2022)


## Appendix 1 : R Code

```
# Advanced Analytics - Classification
#############################################################
# Use Wisconsin Brest Cancer dataset in order to make predictions
# on the diagnosis of patient breats cancer based on Fine Needle
# Aspirate (FNA) method
#############################################################
setwd("F:Advanced Analytics and Machine Learning/Classification")
getwd()
############ Install Packages ##################
install.packages("pacman")
#Pacman allows install all librarys without writing library over and over
pacman::p_load(data.table, 
               fixest, 
               BatchGetSymbols, 
               finreportr, 
               ggplot2, 
               lubridate,
               readxl,
               dplyr,
               tidyverse,
               extrafont,
               ggthemes,
               RColorBrewer,
               scales)
install.packages("readxl")
library(readxl)
install.packages("readr")
library(readr)
set.seed(123)
install.packages("caret")
install.packages("caret", dependencies=T)
library(caret)
install.packages("e1071")
library(e1071)
install.packages("caret")
library(caret)
data <- read_csv("data.csv")
data <- data[2:32] # this code replaces the need for next 2 lines of code
# data$...33 <- NULL
# data$id <- NULL
summary(data)
################# Standardize the data #######################
var(data[,1])
var(data[,2])
stdData <- data %>%
  mutate_if(is.numeric, scale)
# check standardised data
var(stdData[,1])
var(stdData[,2])
############ change diagnosis to binary ####################
stdData$diagnosis[stdData$diagnosis == "M"] <- "1"
stdData$diagnosis[stdData$diagnosis == "B"] <- "0"
stdData$diagnosis <- as.factor(stdData$diagnosis)
summary(stdData$diagnosis)
#Split the data into Test & Train
index <- createDataPartition(stdData$diagnosis,p=0.8,list= F)
train <- stdData[index, ]
test <- stdData[-index, ]
########## Insights & Data Exploration
summary(data)
### CHI Sq Test ###
chisq.test(data$diagnosis, data$radius_mean            )
chisq.test(data$diagnosis, data$texture_mean           )
chisq.test(data$diagnosis, data$perimeter_mean         )
chisq.test(data$diagnosis, data$area_mean              )
chisq.test(data$diagnosis, data$smoothness_mean        )
chisq.test(data$diagnosis, data$compactness_mean       )
chisq.test(data$diagnosis, data$concavity_mean         )
chisq.test(data$diagnosis, data$`concave points_mean`  )
chisq.test(data$diagnosis, data$symmetry_mean          )
chisq.test(data$diagnosis, data$fractal_dimension_mean )
chisq.test(data$diagnosis, data$radius_se              )
chisq.test(data$diagnosis, data$texture_se             )
chisq.test(data$diagnosis, data$perimeter_se           )
chisq.test(data$diagnosis, data$area_se                )
chisq.test(data$diagnosis, data$smoothness_se          )
chisq.test(data$diagnosis, data$compactness_se         )
chisq.test(data$diagnosis, data$concavity_se           )
chisq.test(data$diagnosis, data$`concave points_se`    )
chisq.test(data$diagnosis, data$symmetry_se            )
chisq.test(data$diagnosis, data$fractal_dimension_se   )
chisq.test(data$diagnosis, data$radius_worst           )
chisq.test(data$diagnosis, data$texture_worst          )
chisq.test(data$diagnosis, data$perimeter_worst        )
chisq.test(data$diagnosis, data$area_worst             )
chisq.test(data$diagnosis, data$smoothness_worst       )
chisq.test(data$diagnosis, data$compactness_worst      )
chisq.test(data$diagnosis, data$concavity_worst        )
chisq.test(data$diagnosis, data$`concave points_worst` )
chisq.test(data$diagnosis, data$symmetry_worst         )
chisq.test(data$diagnosis, data$fractal_dimension_worst)
### T-Test ###
t.test(data$radius_mean            ~ data$diagnosis, data=data)
t.test(data$texture_mean           ~ data$diagnosis, data=data)
t.test(data$perimeter_mean         ~ data$diagnosis, data=data)
t.test(data$area_mean              ~ data$diagnosis, data=data)
t.test(data$smoothness_mean        ~ data$diagnosis, data=data)
t.test(data$compactness_mean       ~ data$diagnosis, data=data)
t.test(data$concavity_mean         ~ data$diagnosis, data=data)
t.test(data$`concave points_mean`  ~ data$diagnosis, data=data)
t.test(data$symmetry_mean          ~ data$diagnosis, data=data)
t.test(data$fractal_dimension_mean ~ data$diagnosis, data=data)
t.test(data$radius_se              ~ data$diagnosis, data=data)
t.test(data$texture_se             ~ data$diagnosis, data=data)
t.test(data$perimeter_se           ~ data$diagnosis, data=data)
t.test(data$area_se                ~ data$diagnosis, data=data)
t.test(data$smoothness_se          ~ data$diagnosis, data=data)
t.test(data$compactness_se         ~ data$diagnosis, data=data)
t.test(data$concavity_se           ~ data$diagnosis, data=data)
t.test(data$`concave points_se`    ~ data$diagnosis, data=data)
t.test(data$symmetry_se            ~ data$diagnosis, data=data)
t.test(data$fractal_dimension_se   ~ data$diagnosis, data=data)
t.test(data$radius_worst           ~ data$diagnosis, data=data)
t.test(data$texture_worst          ~ data$diagnosis, data=data)
t.test(data$perimeter_worst        ~ data$diagnosis, data=data)
t.test(data$area_worst             ~ data$diagnosis, data=data)
t.test(data$smoothness_worst       ~ data$diagnosis, data=data)
t.test(data$compactness_worst      ~ data$diagnosis, data=data)
t.test(data$concavity_worst        ~ data$diagnosis, data=data)
t.test(data$`concave points_worst` ~ data$diagnosis, data=data)
t.test(data$symmetry_worst         ~ data$diagnosis, data=data)
t.test(data$fractal_dimension_worst~ data$diagnosis, data=data)
### Visualisations
par(mfrow=c(2,5))
boxplot(data$radius_mean, col="cyan",outcol="red")
boxplot(data$texture_mean, col="cyan",outcol="red")
boxplot(data$perimeter_mean, col="cyan",outcol="red")
boxplot(data$area_mean, col="cyan",outcol="red")
boxplot(data$smoothness_mean, col="cyan",outcol="red")
boxplot(data$compactness_mean, col="cyan",outcol="red")
boxplot(data$concavity_mean, col="cyan",outcol="red")
boxplot(data$`concave points_mean`, col="cyan",outcol="red")
boxplot(data$symmetry_mean, col="cyan",outcol="red")
boxplot(data$fractal_dimension_mean, col="cyan",outcol="red")
plot(data$radius_mean ~ data$area_mean)
### Pair wise plot of all mean variables
pairData <- data[2:6]
pairData1 <- data[7:11]
pairs(pairData, col="blue")
pairs(pairData1, col="blue")
cor.test(data$radius_mean, data$perimeter_mean)
cor.test(data$radius_mean, data$smoothness_mean)
cor.test(data$radius_mean, data$`concave points_mean`)
cor.test(data$radius_mean, data$area_mean)
cor.test(data$`concave points_mean`, data$symmetry_mean)
cor.test(data$perimeter_mean, data$compactness_mean)
############ Clean Data ####################
outlier_radius_mean             <-     boxplot(data$radius_mean            ,na=T)$out
outlier_texture_mean            <-     boxplot(data$texture_mean           ,na=T)$out
outlier_perimeter_mean          <-     boxplot(data$perimeter_mean         ,na=T)$out
outlier_area_mean               <-     boxplot(data$area_mean              ,na=T)$out
outlier_smoothness_mean         <-     boxplot(data$smoothness_mean        ,na=T)$out
outlier_compactness_mean        <-     boxplot(data$compactness_mean       ,na=T)$out
outlier_concavity_mean          <-     boxplot(data$concavity_mean         ,na=T)$out
outlier_concave_points_mean     <-     boxplot(data$`concave points_mean`  ,na=T)$out
outlier_symmetry_mean           <-     boxplot(data$symmetry_mean          ,na=T)$out
outlier_fractal_dimension_mean  <-     boxplot(data$fractal_dimension_mean ,na=T)$out
sum(outlier_radius_mean           >0,na=T)
sum(outlier_texture_mean          >0,na=T)
sum(outlier_perimeter_mean        >0,na=T)
sum(outlier_area_mean             >0,na=T)
sum(outlier_smoothness_mean       >0,na=T)
sum(outlier_compactness_mean      >0,na=T)
sum(outlier_concavity_mean        >0,na=T)
sum(outlier_concave_points_mean   >0,na=T)
sum(outlier_symmetry_mean         >0,na=T)
sum(outlier_fractal_dimension_mean>0,na=T)
summary(outlier_area_mean)
data[data$radius_mean            %in% outlier_radius_mean, "radius_mean"] = NA
data[data$texture_mean           %in% outlier_texture_mean, "texture_mean"] = NA
data[data$perimeter_mean         %in% outlier_perimeter_mean, "perimeter_mean"] = NA
data[data$area_mean              %in% outlier_area_mean, "area_mean"] = NA
data[data$smoothness_mean        %in% outlier_smoothness_mean, "smoothness_mean"] = NA
data[data$compactness_mean       %in% outlier_compactness_mean, "compactness_mean"] = NA
data[data$concavity_mean         %in% outlier_concavity_mean, "concavity_mean"] = NA
data[data$`concave points_mean`  %in% outlier_concave_points_mean, "concave points_mean"] = NA
data[data$symmetry_mean          %in% outlier_symmetry_mean, "symmetry_mean"] = NA
data[data$fractal_dimension_mean %in% outlier_fractal_dimension_mean, "fractal_dimension_mean"] = NA
# data na.omit(data) # Decision to keep values as NA
summary(data)
### Clean data Visualisations
par(mfrow=c(2,5))
boxplot(data$radius_mean, col="cyan",outcol="red")
boxplot(data$texture_mean, col="cyan",outcol="red")
boxplot(data$perimeter_mean, col="cyan",outcol="red")
boxplot(data$area_mean, col="cyan",outcol="red")
boxplot(data$smoothness_mean, col="cyan",outcol="red")
boxplot(data$compactness_mean, col="cyan",outcol="red")
boxplot(data$concavity_mean, col="cyan",outcol="red")
boxplot(data$`concave points_mean`, col="cyan",outcol="red")
boxplot(data$symmetry_mean, col="cyan",outcol="red")
boxplot(data$fractal_dimension_mean, col="cyan",outcol="red")
########### Best Subset Selection ####################
x <- model.matrix(diagnosis ~ . -1, data=stdData) # '-1' removes the intercept
y <- stdData$diagnosis
library(leaps)
bestmods <- leaps(x,y,nbest=1)
bestmods
bestmods$Cp
min(bestmods$Cp)
###
regfit.full <- regsubsets(diagnosis ~ ., data = stdData,
                          nvmax = 20)
reg.summary <- summary(regfit.full)
###
names(reg.summary)
###
reg.summary$rsq
reg.summary
dev.off()
par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables",
     ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "Number of Variables",
     ylab = "Adjusted RSq", type = "l")
which.max(reg.summary$adjr2)
points(11, reg.summary$adjr2[11], col = "red", cex = 2, 
       pch = 20)
plot(reg.summary$cp, xlab = "Number of Variables",
     ylab = "Cp", type = "l")
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col = "red", cex = 2,
       pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic, xlab = "Number of Variables",
     ylab = "BIC", type = "l")
points(6, reg.summary$bic[6], col = "red", cex = 2,
       pch = 20)
###
dev.off()
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
###
which.min(reg.summary$bic)
coef(regfit.full, 11) #model with lowest BIC is the 11 variable model
##### SVM with Multiple Classes ####################
classifier = svm(formula = diagnosis ~ .,
                 data = train,
                 type = 'C-classification',
                 kernel = 'linear')
classifier
classifier$coefs
svmfit <- svm(diagnosis ~ .,
              data = train,
              kernel = "radial", 
              cost = 10,
              gamma = 1)
plot(svmfit, data[train,])
plot(svmfit, train)
summary(svmfit)
y_pred = predict(classifier, newdata = test)
y_pred1 = predict(svmfit, newdata = test)
predict(svmfit, newdata = test)
y_pred
y_pred1
table(y_pred1)
cm = table(test$diagnosis,y_pred1)
cm
pred <- predict(model, x, decision.values = TRUE, probability = TRUE)
set.seed(123)
tune.out <- tune(svm, diagnosis ~ ., data = train, 
                 kernel = "radial", 
                 ranges = list(
                   cost = c(0.1, 1, 10, 100),
                   gamma = c(0.5, 1, 2, 3, 4)
                 )
)
summary(tune.out)
pred <- prediction(as.numeric(pred),as.numeric(stdData$diagnosis))
### Error - cannot get an output from below code
### Error - Subscript `-train` has the wrong type `data.frame<
# table(
#   diagnosis = stdData[,1], 
#   pred = predict(
#     tune.out$best.model, newdata = stdData[,]
#   )
# )
# 
# ## How can below code provide accurate data
# predict(tune.out$best.model, newdata = test)
# table(predict(tune.out$best.model, newdata = test))
############## Logistic Regression ##############
data$diagnosis <-  as.numeric(data$diagnosis)
###
glm.fits <- glm(diagnosis ~ radius_mean + texture_mean + perimeter_mean +
                  area_mean +smoothness_mean + compactness_mean +
                  concavity_mean +`concave points_mean` + symmetry_mean +
                  fractal_dimension_mean,data = data, family = binomial)
glm.fits <- glm(diagnosis ~ radius_mean + perimeter_mean +
                 area_worst,
                data = data, family = binomial)
glm.fits <- glm(diagnosis ~ area_mean+`concave points_mean`+texture_worst+
                  compactness_se,
                data = data, family = binomial)
summary(glm.fits)
train$y_pred = predict(glm.fits, train, type="response")
###
coef(glm.fits)
summary(glm.fits)$coef
summary(glm.fits)$coef[, 4] #
summary(glm.fits)$coef[, 3] #can be changed to 3 to give the Z value
###
glm.probs <- predict(glm.fits, type = "response")
glm.probs[1:10] ## What does this code tell us?
mean(glm.probs)
glmSubset <- glm(diagnosis ~ compactness_mean + concavity_mean + radius_se +
                   smoothness_se + concavity_se + radius_worst + texture_worst +
                   area_worst + `concave points_worst` + symmetry_worst + 
                   fractal_dimension_worst,
                 data = train,
                 family = binomial)
summary(glmSubset)
train$y_pred = predict(glmSubset, train, type = "response")
train$y_pred
prob <- glmSubset %>%
  predict(test, type = "response")
head(prob)
pred.classes <- ifelse(prob > 0.5, "1", "0")
head(pred.classes)
mean(pred.classes == test$diagnosis)
test$diagnosis <- as.factor(test$diagnosis)
train$diagnosis <- as.factor(train$diagnosis)
summary(test$diagnosis)
summary(train$diagnosis)
glmProb <- predict(glmSubset, type = "response")
glmProb[1:20]
contrasts(train$diagnosis)
glmPred <- rep("1", 455)
glmPred[glmProb > 0.5] = "2"
### Logistic Reg Preds
table(glmPred, train$diagnosis)
mean(glmPred == 1)
glmProbTest <- predict(glmSubset, test, type = "response")
glmPred1 <- rep("1", 113)
glmPred1[glmProbTest > 0.5] = "2"
table(glmPred1, test$diagnosis)
mean(glmPred1 == 1)
############## Linear Discriminant Analysis  ###############
install.packages("MASS")
library(MASS)
library(ipred)
library(tidyverse)
library(caret)
set.seed(123)
# Fit the model
model <- lda(diagnosis ~., data = train)
plot(model)
# Make predictions
pred <- predict(model, train)
predClass <- pred$class
table(predClass, train$diagnosis)
predA <- predict(model, test)
predAClass <- predA$class
table(predAClass, test$diagnosis)
# Model accuracy
mean(pred$class==train$diagnosis)
mean(predA$class==test$diagnosis)
model1 <- lda(diagnosis ~ compactness_mean+concavity_mean+radius_se+
               smoothness_se+concavity_se+radius_worst+texture_worst+
               area_worst+`concave points_worst`+symmetry_worst+fractal_dimension_worst,
             data = train)
plot(model1)
# Make predictions
pred1 <- predict(model1, train)
pred1Class <- pred1$class
table(pred1Class, train$diagnosis)
predA1 <- predict(model1, test)
predA1Class <- predA1$class
table(predA1Class, test$diagnosis)
# Model accuracy
mean(pred1$class==train$diagnosis)
mean(predA1$class==test$diagnosis)
#################### ensemble ####################
#Define training controls
fitControl <- trainControl(method="cv",
                           number=5,
                           savePredictions = 'final',
                           classProbs = T)
#Define predictors & outcomes
predictors <- c("compactness_mean","concavity_mean","radius_se",
                "smoothness_se","concavity_se","radius_worst","texture_worst",
                "area_worst","concave points_worst","symmetry_worst","fractal_dimension_worst")
outcomeName <- 'diagnosis'
str(outcomeName)
outcomeName <- as.factor(outcomeName)
#Random Forest ## Unfinished - issue with outcome class column
test$diagnosis <- as.numeric(test$diagnosis)
train$diagnosis <- as.numeric(train$diagnosis)
train <- as.data.frame(train)
model_rf <- train(train[,predictors],
                  train[,outcomeName],
                  methods='rf',
                  trControl=fitControl,
                  tuneLength=3)
test$pred_rf <- predict(object=model_rf,test[,predictors])
pred <- predict(object=model_rf,test[,predictors])
confusionMatrix(test$diagnosis,test$pred_rf)
str(pred)
### Bagging
library(dplyr)       #for data wrangling
library(e1071)       #for calculating variable importance
library(caret)       #for general model fitting
library(rpart)       #for fitting decision trees
library(ipred)       #for fitting bagged decision trees
#view structure dataset
str(data)
#make this example reproducible
set.seed(123)
#fit the bagged model
bag <- bagging(
  formula = diagnosis ~ .,
  data = data,
  nbagg = 100,   
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)
#display fitted bagged model
bag
#calculate variable importance
VI <- data.frame(var=names(data[,-1]), imp=varImp(bag))
summary(bag)
#sort variable importance descending
VI_plot <- VI[order(VI$Overall, decreasing=TRUE),]
### Boosting
install.packages("xgboost")
library(xgboost) #for fitting the xgboost model
library(caret)   #for general data preparation and model fitting
#Note that the xgboost package also uses matrix data,
#so we’ll use the data.matrix() function to hold our predictor variables.
set.seed(123)
#define predictor and response variables in training set
train_x = data.matrix(train[, -1])
train_x = as.matrix.data.frame(train[, -1])
train_y = train[,1]
#define predictor and response variables in testing set
test_x = data.matrix(test[, -1])
test_x = as.matrix.data.frame(test[, -1])
test_y = test[, 1]
#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)
#define watchlist
watchlist = list(train=xgb_train, test=xgb_test)
#fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 70)
#################### Neural Network ####################
train$diagnosis <- as.factor(train$diagnosis)
install.packages("neuralnet")
library(neuralnet)
nMod <- neuralnet(diagnosis~ radius_mean+smoothness_mean+perimeter_se,
                  data = stdData,
                  hidden=1,
                  threshold=0.01,
                  learningrate.limit = NULL,
                  learningrate.factor = list(minus=0.5,plus=1.2),
                  algorithm="rprop+")
nMod$result.matrix
plot(nMod)
nMod$result.matrix
pred <- predict(nMod, test)
table(test$diagnosis == "1",pred[,1]>0.5)
predA <- predict(nMod, train)
table(train$diagnosis == "1",predA[,1]>0.5)
### Use the subset selection features to create a NN 
nMod1 <- neuralnet(diagnosis ~ compactness_mean+concavity_mean+radius_se+
                     smoothness_se+concavity_se+radius_worst+texture_worst+
                     area_worst+symmetry_worst+fractal_dimension_worst,
                  data = stdData,
                  hidden=1,
                  threshold=0.01,
                  learningrate.limit = NULL,
                  learningrate.factor = list(minus=0.5,plus=1.2),
                  algorithm="rprop+")
nMod1$result.matrix
pred1 <- predict(nMod1, test)
table(test$diagnosis == "1",pred1[,1]>0.5)
pred1A <- predict(nMod1, train)
table(train$diagnosis == "1",pred1A[,1]>0.5)
### Neural Net End 
#########################################################################

```


