# Backorder

## Introduction :
A Backorder is an order which can’t be fulfilled at the given time due to lack of supply or the product is currently out of stock or not in inventory but can guarantee delivery of the goods or service requested by a certain date in the future because the production of goods or replenishment of inventory is underway. Unlike in the situation of Out-of-stock where the delivery date of the goods can’t be promised , in the Backorder scenario the customers are allowed to shop for the products and order. Simply put Backorder can be thought of as an order with a delayed delivery date.

## Why do Backorders happen ?
1. <b><u>When there is a sudden increase in demand : </u></b> The primary goal of all the companies is to increase the demand for the products they offer. Having a poor sales forecast system could one of the reasons for failing to predict the demand. Despite having a good sales forecasting system sometimes these situations are inevitable because of the factors which can’t be controlled or un predictable events.
2. <b><u>Poor Supply chain Management :</u></b> If there is a breakdown at any point in the supply chain or due to improper planning a stockout scenario arises affecting the production. Having limited options for sourcing the raw materials and improper vendor management system is one of the significant reasons for Backorder.
3. <b><u>Inventory Management : </b></u>Improper management of inventory operations and not having visibility of the inventory storage could lead to Backorders.

## Effects of Backorders:
1. If many items are going on Backorders consistently it is a sign that companies operations are not properly planned and also there is a very high chance of missing out business on the products.
2. Also if the customers frequently experience Backorders they switch their loyalities to your competitors.
3. Backorders(unpredicted) also affect the production planning, Transportation management and logistics management etc.

## What to do to avoid Backorders ?
1. Increasing inventory or stock of produts is not a solution as it increases storage costs and extra costs means they have to be included in the product prices which might result in losing business to competitors.
2. A well planned Suppply Chain Management , Warehouse management and inventory management can avoid Backorders to some extent.

## Need for Having Backorder prediction system:
1. Backorders are inevitable but through prediction of the items which may go on backorder planning can be optimized at different levels avoiding un expected burden on production , logistics and transportation planning.
2. ERP systems produce a lot of data (mostly structured) and also would have a lot of historical data , if this data can be leveraged correctly a Predictive model can be developed to forecast the Backorders and plan accordingly.

## Problem Statement :
Classify the products whether they would go into Backorder(Yes or No) based on the historical data from inventory, supply chain and sales.

## Presenting as an ML problem:
The task at hand is classifying whether a product would go to Backorder given input data. The target variable to predict consists of two values:
1. “Yes” - If the Product predicted to go to Backorder.
2. “No”- If the Product predicted to be not going to Backorder.\
So it is a Binary Classification problem.


## Dataset Analysis :
In the Train dataset we are provided with 23 columns(Features) of data.
1. Sku(Stock Keeping unit) : The product id — Unique for each row so can be ignored
2. National_inv : The present inventory level of the product
3. Lead_time : Transit time of the product
4. In_transit_qty : The amount of product in transit
5. Forecast_3_month , Forecast_6_month , Forecast_9_month : Forecast of the sales of the product for coming 3 , 6 and 9 months respectively
6. Sales_1_month , sales_3_month ,sales_6_month , sales_9_month : Actual sales of the product in last 1 , 3 ,6 and 9 months respectively
7. Min_bank : Minimum amount of stock recommended
8. Potential_issue : Any problem identified in the product/part
9. Pieces_past_due: Amount of parts of the product overdue if any
10. Perf_6_month_avg , perf_12_month_avg : Product performance over past 6 and 12 months respectively
11. Local_bo_qty : Amount of stock overdue
12. Deck_risk , oe_constraint, ppap_risk, stop_auto_buy, rev_stop : Different Flags (Yes or No) set for the product
13. Went_on_backorder : Target variable\

The class ratio of Products that went to Backorder(‘Yes’) to those which didn’t go to Backorder(‘No’) is 1:148. The dataset is highly imbalaced which should be addressed for accurate predictions by the model. Out of the 23 features given in the dataset 15 are numerical and 8(including the target variable) are categorical features. The first column ‘sku’ corresponds to product identifier which is unique for each datapoint in the dataset. So this feature can be dropped as it adds no value in output prediction.

## Data Cleaning:
In both training and test dataset features such as Lead_time, Perf_6_month_avg, Perf_12_month_avg contains NaN values. So, in order to clean the data we used 
the centeral tendency value Median. The important point in Data Cleaning is that features Perf_6_month_avg, Perf_12_month_avg contains vlaue -99.0 which we first 
converted into np.nan and then we applied the fillna() function to complete our dataset.

![1](https://user-images.githubusercontent.com/41980059/132715326-3d1cab18-2282-4a56-b2ba-d2b2b1357028.png)

## Exploratory Dataset Analysis :
By now, we know that our dataset is highly imbalanced and we can see it clearly when we plot bar graphs for numerical fetaures and count plot for categorical fetaures.
When we plot Box Plot for observing the distribution of various numerical features. On first looking the plots we can conclude that there are a lot of Outliers in out dataset.
But coming to this conclusion on the first look and removing these outliers can lead to loosing some important information from our dataset.

![2](https://user-images.githubusercontent.com/41980059/132716776-be79088a-8f1c-4669-a8ef-2554accfce89.png)
![3](https://user-images.githubusercontent.com/41980059/132717448-05f924bb-9765-4a4a-811c-08d67e1b5d3e.png)

## Feature Engineering :
Backorder dataset has very similar named columns. So,before moving to Feature Engineering and Feature Selection we checked how much the features correlates to each other.
Because if feature values have high correlation values then it is redundant to keep all the values else we should do dome feature engineering using these correlated features.


![3](https://user-images.githubusercontent.com/41980059/132719247-59c60098-93c7-44f4-836f-e4253e545137.png)


Forecast_3_month, forecast_6_month,forecast_9_month are highly correlated to one another. In_transit_qty,sales_3_month and min_bank are correlated to each other. sales_1_month,sales_3_month,sales_6_month and sales_9_month are highly correlated. perf_6_month_avg,perf_12_month_avg are highly correlated to each other.
So we can use Feature engeneering in order to remove the multicolliniearity in our dataset.

![4](https://user-images.githubusercontent.com/41980059/135072681-5a5d81ab-ac58-4a9a-9e63-7f642ec7a6e3.png)

## Feature Scaling :
Since the features in our dataset have different different scales. So, one particular fetaure can affect our final value more than other features. In order to stop this to happen we use Standard Scaler which basically sclaes the feartures at the same lvel so that all features have same weightage.

## Modelling :
Since our dataset is highly imbalanced there are various ML models which can handle the imbalance nature of dataset by itself without the need of any other data cleaning process. Such ML models are like KNN classifier, Bagging Classifier, Balanced bagging Classifier, Random Forest Classifier, Random Forest Classifier with weights, Easy Ensemble Classifier, Gradient Boosting Classifier.

Since recall value is the most important metric for the given statement and easyensembleclassifier was the one with the best recall value of 85%. 
![4](https://user-images.githubusercontent.com/41980059/135075647-0ae5a797-b4be-4ef5-a62f-baf2ef51f140.png)
