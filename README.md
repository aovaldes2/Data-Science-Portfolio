# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks.

_Note: Data used in the projects (accessed under data directory) is for demonstration purposes only._

## Contents

- ### Machine Learning
  - ### Supervised Learning

	- [Of Genomes And Genetics](https://github.com/aovaldes2/-Data-Science-Portfolio/tree/main/Of%20Genomes%20And%20Genetics/notebooks): A model to predict  children's Genetic disorders with the Disorder subclass. The aim is to identify a reliable model for the genetic diagnosis using machine learning ([Details](#of-genomes-and-genetics)). 
	- [Forest Cover Type Prediction](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/Forest_Cover_Type_Prediction.ipynb): The goal of the Project (and competition): to predict seven different cover types in four different wilderness areas of the Roosevelt National Forest of Northern Colorado with the best accuracy([Details](#forest-cover-type-prediction)). 
	- [New York City Taxi-Trip Duration](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/New_York_City_Taxi_Trip_Duration.ipynb): Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City ([Details](#new-york-city-taxi-trip-duration)). 
	
	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib_

  - ### Natural Language Processing

	- [Quora Insincere Questions Classification](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/Quora_Insincere_Questions.ipynb): Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
In this project we will develop models that identify and flag insincere questions.

	_Tools: Pandas, Seaborn, Matplotlib, NLTK, scikit_

- ### Data Analysis and Visualisation
	- __Exploratory Data Analysis projects__ 
		
		- [Madrid House Price](https://github.com/aovaldes2/-Data-Science-Portfolio/blob/main/Madrid%20House%20Price%20EDA/Madrid_RealState.ipynb): Analysis of the different factors that can influence the price of houses in Madrid.
	
	_Tools: Pandas, Seaborn and Matplotlib_

------------------------------------------------------------------------------------------------------------------------------------------------------------
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
# Forest Cover Type Prediction <a id='forest-cover-type-prediction'></a>

	
In this Kaggle competition ([link](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview)), we are asked to predict forest cover type from strictly cartographic variables. (USFS). The data is in raw (unscaled) format and contains binary columns of data for qualitative independent variables, such as wilderness and soil type.

The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

1 - Spruce/Fir

2 - Lodgepole Pine

3 - Ponderosa Pine

4 - Cottonwood/Willow

5 - Aspen

6 - Douglas-fir

7 - Krummholz

The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).


In this case I use a single notebook loadable from google colab.

First, the corresponding EDA is carried out. In this EDA, the great variability of the data comes to light from distance, grades, indices to categorical variables that represent the existence or not of different types of soils in the different Wilderness Areas (which also represent a variable).
It can also be seen how the target variable is well distributed in the training data (2160 each). 


Two types of univariate and bivariate analyzes are performed in the EDA. 

#### Univariate Analysis resume:

* Train dataset has 15120 rows and 56 columns.
* Each column has numeric (integer/float) datatype.
* There are no NA in the dataset.Thus dataset is properly formatted and balanced.
* Only 4 columns had outliers.
    1. Horizontal_Distance_To_Hydrology
    2. Vertical_Distance_To_Hydrology
    3. Horizontal_Distance_To_Roadways
    4. Horizontal_Distance_To_Fire_Points
* Cover_Type is our label/target column

#### Bivariate Analysis resume:

* The importance of the categorical variable Wilderness Area Type (created from the variables Wilderness_Area#) is verified by analyzing the density and distribution of the values with respect to the values of our target variable Cover_Type.
* The Relation and Distribution of continuous variables (Elevation, Aspect, Slope, Distance and Hillsahde columns) were analyzed in addition to the  highly correlated features such as:
	1. hillshade noon - hillshade 3pm
	2. hillshade 3 pm - hillshade 9 am
	3. vertical distance to hydrology - horizontal distance to hydrology
	4. elevation - slope

Some high correlations between hillshade variables, distance to hydrology. Makes sense since these variables seem interrelated.


* The pearson coefficients showed that none of the base features have significantly linear effect in determining the label cover type. In addition, one interesting finding was that Soil Type 7 and 15 correlation are none in the pearson table, or what is the same it has no effect on determining the label Cover_Type according to the data.

#### Baseline Results (No Feature Engineering)
In this section, the results of different models would be analyzed with the features without Feature Engineering to select a model in the first instance.

![img1FCTP](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/images/i0Comparations%5BFCTP%5D.png)

Extreme (extra) random forests outperformed other algorithms with better accuracy performance in this case. The reason might be, I did not focus on tuning the parameters of the each algorithm and used defaults values instead. In any case, due to the type of problem that arises, we will focus on the feature engineering and hyperparameter tuning.

This baseline submision score 0.74071 in the Kaggle Competition

#### Feature Engineering & Selection

 Since the test data is much larger than the training data, and performs differently, I'm not going to remove any predictors. Instead I'm going to focus on creating new ones that highlight similarities in the data. Due to the characteristics of some features, it would make sense to introduce new ones and observe their implication in the models.
 
 Feature engineering was separated into blocks to compare their improvement in the scores. Engineering was used from the simplest of adding columns or having maxima to some more complex ones such as adding absolutes or Euclidean distances to certain points, always based on the characteristics of the features themselves.
 
 Adding a few at a time and then checked the results on the validation data we can see especially for the extra trees model, accuracy kept increasing as predictors were added.
 
 #### Perform Hyperparameter Tuning on the Bests Models

As we know optimizing the hyperparameters for machine learning models is vital to the performance of the machine learning models and the process of tuning the HPs is not intuitive and can be a complex task. In this case its use Randomized Search for a shorter run-time although may not return the best combination of hyper-parameters that would return the best accuracy, this method doesn't consider past evaluations and it will continue the iterations regardless of the results. In the end, for the chosen model, it was obtained:

```
Best ExtraTreesClassifier Params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None}
```
Also as part of the experimentation, the parameters of a Light Gradient Boosting Machine classifier model were tuned with the same predictors mentioned above:
```
Best LGBMClassifier Params: {'num_leaves': 31, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.6}
```
In the following image we can see the best submissions where it can be noted that in the entries written in the form Name(1) all the predictors were already included but as part of the experimentation in the first instance there were fewer created features.

![img1FCTP](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/images/Submisions%5BFCTP%5D.png)


--------------------------------------------------------------------------------
# New York City Taxi-Trip Duration <a id='new-york-city-taxi-trip-duration'></a>


In this competition, Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City. Your primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should predict the duration of each trip in the test set.

File descriptions:

* train.csv - the training set (contains 1458644 trip records)
* test.csv - the testing set (contains 625134 trip records)

In this case I use a single notebook loadable from google colab.

In the first instance, a simple EDA is carried out with the base features of the problem, where it is observed that there are no Nan elements in the database, that this database has features such as pickup and dropoff time(presented as datetime) as well as the respective coordinates of the trip (each presented as a float).
Other types of data such as object or integers its found in the rest of the features such as _passenger_count, store_and_fwd_flag, vendor_id_. The target feature _trip_duration_ was presented as an integer since it represents the time in seconds.

#### Feature Creation(Feature Engineering & Selection)


##### Distances
New features were created from existing ones, for example, the datetimes were broken down to the unit of minutes, as well as the coordinates of the trip were used to calculate 2 types of distances,**Haversine Distance** and **Dummy Distance**.

> **Haversine Distance** - Euclidean Distance works for the flat surface like a Cartesian plain however, Earth is not flat. So we have to use a special type of formula known as Haversine Distance. Haversine Distance can be defined as the angular distance between two locations on the Earth's surface.

> **Dummy Distance** - The distance calculated from Haversine Distance made up of the perpendicular paths between the two points in question (Let P1(lat1, lng1) and P2(lat2, lng2) then dummy.distance = haversine.distance(lat1, lng1, lat1, lng2)+haversine.distance(lat1, lng1, lat2, lng1)).

These new features gave the possibility of calculating the average speed of each trip as well as its breakdown by hours, day of the week, etc.

![img1NYCTD](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/images/SpeedDesc%5BNYCTD%5D.png)

##### Direction
Another important feature created in this section will be the direction of the trip, this importance could be seen graphically by representing the differences between latitudes and longitudes:

![img2NYCTD](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/images/DirIntu%5BNYCTD%5D.png)

##### Clustering
Besides of keeping entire list of latitude and longitute, the data will be grouped by some approximate locations. It might be helpful for tree-based algorithms.

#### Additional Datasets

##### OSMR features
We had only rough distance estimates in the previous versions. We will now use additional data extracted from OpenStreetMap which was used successfully in the top scores. Most of the high scores use [Data about Fastest Routes](https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm). Travel distance should be more relevent here. The difficult part is to adquire this feature. Thanks to Oscarleo who manage to pull it off from OSRM.

New features were added to our model from this database, such as _total_distance,total_travel_time_ and _number_of_steps_.


##### Weather features

This dataset([KNYC Metars 2016](https://www.kaggle.com/datasets/cabaki/knycmetars2016)) is ment to be used as a extra information for those willing to extract conclusions from their own dataset where hourly weather information could be useful for their predictions / analysis. This is the METARs aggregated information for 2016 in KNYC. In this case where added _minimum temperature, precipitation, snow fall_ and _snow depth_ to our model.


#### Model Training(Hyperparameter Tuning)


Several algorithms can be used in this case I prefer a tree-based algorithm XGBRegressor. As we know optimizing the hyperparameters for machine learning models is vital to the performance of the machine learning models and the process of tuning the HPs is not intuitive and can be a complex task. In this case its use Randomized Search for a shorter run-time although may not return the best combination of hyper-parameters that would return the best score(neg_mean_squared_error), this method doesn't consider past evaluations and it will continue the iterations regardless of the results. In the end, for the chosen model, it was obtained:

```
<bound method XGBModel.get_xgb_params of XGBRegressor(booster='gblinear', colsample_bytree=0.6, eta=0.04,
             eval_metric='rmse', gamma=2, lambda=2, max_depth=10,
             min_child_weight=5, n_estimators=500, nthread=-1, predictor='u',
             random_state=42, silent=1, subsample=0.75)>
```

![img1FCTP](https://github.com/aovaldes2/Data-Science-Portfolio/blob/main/images/Submision%5BNYTD%5D.png)


----------------------------------------------------------------------------------------------------------------------------------------------------------------
If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at aovaldes2@gmail.com. 


### Support My Work

_If this project inspired you, gave you ideas for your own portfolio or helped you, please consider [buying me a coffee](https://www.buymeacoffee.com/aovaldes2) ❤️._ 
