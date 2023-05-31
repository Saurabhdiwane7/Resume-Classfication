# Resume-Classfication

![resumes](https://tse4.mm.bing.net/th?id=OIP.rXPL79KdYF7G-tZzfuWfzwHaEc&pid=Api&P=0&h=180)

Abstract:

This project aims to develop a machine learning-based approach for automated resume classification. By leveraging natural language processing techniques, resumes can be efficiently analyzed and categorized into predefined job categories. This automation streamlines the recruitment process, saving time and effort for recruiters while ensuring a more systematic and objective evaluation of candidate profiles.

Introduction:

Resume classification plays a vital role in the recruitment process by efficiently categorizing resumes to identify suitable candidates for job positions. This project focuses on developing a machine learning model that uses natural language processing techniques to automatically extract information from resumes and classify them into predefined categories. This automation improves the efficiency and objectivity of candidate evaluation, enabling recruiters to make more informed decisions.

## Objective 

♦ Objective for above project is to classify resumes and reduce the manual human effort in the HRM. 

## Data Summary

♦ Given dataset contain four folders of resumes category like Peoplesoft Resume , Workday Resume, React JS Developer Resumes and Hexaware Resumes.

Peoplesoft Resume = 20 Resumes

Workday Resumes =14 Resumes

React JS Developer Resumes =23 Resumes

Hexaware Resumes = 21 Resumes


## EDA and Visualizations

♦ Using Functions  we’ve stored all the Resumes in one data frame with column names
‘Resumes , Category’ and proceed with the Cleaning of data by regular expressions using NLTK library.

♦ We’ve removed stopwords ,hashtags ,web links, mentions ,extra whitespaces from the data.
And convert all the data in csv file and proceed for EDA.

♦ We’ve created a function to check the most common keyword used in the resumes and their frequency, also we have lemmatize the similar keyword using Wordnetlemmatizer.

♦ After lemmatizing the resumes filtered column as per the category unique values and we’ve plotted the data distribution with respect to category.

♦ We’ve also plotted the wordclouds for each category to check the frequency of most occurring keywords in each category of resumes.

## Model Building and Model Evaluation

♦ We’ve divided the data using train-test split and used TfidfVectorizer to convert text data into a numerical representation that can be used by machine learning models. It stands for Term Frequency-Inverse Document Frequency and is a way to measure the importance of a word in a document 

♦ Used Random Forest,Logistic Regression, Multinomial Navies Bayes,Support Vector Machine alogrithms

♦ used Adaboosting,Gradient Boosting,XGB,LGB etc. algorithms.

♦ Calculated Accuracy score,Recall Score,Precision Score, F1 scores for above models.

♦ Taken Random Forest as final model as it shows best fit accuracy.

## Model Deployment

♦ We’ve deployed our model in Streamlit web framework.
