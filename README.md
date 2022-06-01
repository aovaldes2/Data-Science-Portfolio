# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks.

_Note: Data used in the projects (accessed under data directory) is for demonstration purposes only._

## Contents

- ### Machine Learning
  - ### Supervised Learning

	- [Of Genomes And Genetics](https://github.com/aovaldes2/-Data-Science-Portfolio/tree/main/Of%20Genomes%20And%20Genetics/notebooks): A model to predict  children's Genetic disorders with the Disorder subclass. The aim is to identify a reliable model for the genetic diagnosis using machine learning ([Details](#of-genomes-and-genetics)). 
	- [Forest Cover Type Prediction](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/Forest_Cover_Type_Prediction.ipynb): The goal of the Project (and competition): to predict seven different cover types in four different wilderness areas of the Roosevelt National Forest of Northern Colorado with the best accuracy.
	- [New York City Taxi-Trip Duration](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/New_York_City_Taxi_Trip_Duration.ipynb): Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City.
	
	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib_

  - ### Natural Language Processing

	- [Quora Insincere Questions Classification](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/Quora_Insincere_Questions.ipynb): Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
In this project we will develop models that identify and flag insincere questions.

	_Tools: Pandas, Seaborn, Matplotlib, NLTK, scikit_

- ### Data Analysis and Visualisation
	- __Exploratory Data Analysis projects__ 
		
		- [Madrid House Price](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/Madrid%20House%20Price%20EDA/Madrid_RealState.ipynb): Analysis of the different factors that can influence the price of houses in Madrid.
	
	_Tools: Pandas, Seaborn and Matplotlib_

----------------------------------------------------------------------------
# Of Genomes And Genetics <a id='of-genomes-and-genetics'></a>

You are given a dataset that contains medical information of children who have genetic disorders. Predict the following:
- Genetic disorder 
- Disorder subclass

With the Evaluation metric

Genetic Disorder
```
score1 = max(0, 100*metrics.f1_score(actual["Genetic Disorder"], predicted["Genetic Disorder"], average="macro"))
```
Disorder Subclass
```
score2 = max(0, 100*metrics.f1_score(actual["Disorder Subclass"], predicted["Disorder Subclass"], average="macro"))
```
Final score
```
score = (score1/2)+(score2/2)
```
I used two main notebooks 

1_GeneticDisorder_preprocessing.ipynb ([link](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/Of%20Genomes%20And%20Genetics/notebooks/1_GeneticDisorder_preprocessing.ipynb))

Although a much more extensive EDA was performed on the notebook named
GeneticDisorderEDAv1 the most important transformations to the data for modeling occurred on this one. Of the 44 columns, 13 that would not provide relevant information were discarded at first, from the first moment it can be observed in the data that they present many missing data to solve this problem, MICE (Multiple Imputations by Chained Equations) imputation will be applied. This approach may be generally referred to as fully conditional specification (FCS) or multivariate imputation by chained equations (MICE).

> This methodology is attractive if the multivariate distribution is a reasonable description of the data. FCS specifies the multivariate imputation model on a variable-by-variable basis by a set of conditional densities, one for each incomplete variable. Starting from an initial imputation, FCS draws imputations by iterating over the conditional densities. A low number of iterations (say 10–20) is often sufficient.

— mice: Multivariate Imputation by Chained Equations in R, 2009.

Non-numeric values can be observed in the data whose transformation is important for our modelling. For this, the variables were separated between categorical and non-categorical to later apply the ordinal encoding method to encode the categorical variables. The variant of using ordinal encoding came from experimentation and obviously these operations were performed by joining the test and training data.

2_GeneticDisorder_finalmodelling.ipynb ([link](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/Of%20Genomes%20And%20Genetics/notebooks/2_GeneticDisorder_finalmodeling.ipynb))

In this notebook the main objective is to perform the runs using different models. For this, the data is also transformed, starting with our target variables: Genetic Disorder and Disorder Subclass, which are categorical variables whose characteristics and the data itself allow us to create a single variable without the risk of overlapping. Then, when analyzing the new target variable, it can be seen that the data is unbalanced. During the experimentation, an improvement in the results was observed when the the target variable is balanced, so the data was balanced by Over Sampling using SMOTE (Synthetic Minority Oversampling TEchnique)

This technique was described by Nitesh Chawla, et al. in their 2002 paper named for the technique titled “SMOTE: Synthetic Minority Over-sampling Technique.”

>SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line. 

Then we proceed to run the models whose results can be seen in the following table:

| Model        | Score           | 
| ------------- |:-------------:| 
| GradientBoostingClassifier |33.60688| 
| XGBoostClassifier |33.52182| 
| LGBMClassifier |32.99259| 
| CatboostClassifier |35.20506| 

The best ranked result would be the CatboostClassifier model in 18th place out of 2138 participants.

--------------------------------------------------------------------------------

If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at aovaldes2@gmail.com. 

### Support My Work

_If this project inspired you, gave you ideas for your own portfolio or helped you, please consider [buying me a coffee](https://www.buymeacoffee.com/aovaldes2) ❤️._ 
