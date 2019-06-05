
# Using NLP Modeling for Aircraft Maintenance

_By Deepa Mahidhara_

Aircrafts have maintenance issues that are identified and written up as maintenance open logs during various stages of the operations.These open logs need to be addressed by the maintenance operations team to keep the aircraft in service. 
Some of the maintenance logs are addressed during daily operations and others are deferred to be addressed at a later time during nightly maintenance. Technicians review these logs to identify the action needed -  this is currently highly dependent on the experience of a technician, and the manual process of referring to a maintenance manual.

PROBLEM STATEMENT
Build a Machine Learning model to:
  - Read maintenance logs
  - Convert them to machine-readable language using NLP techniques
  - Predict an action based on historical data

BENEFITS
  - Speed up the maintenance process
  - Reduce human errors
  - Reduce airline delays & cancellations

During disasters, search and rescue teams need to find the fastest and most effective routes. Currently, live traffic maps are failing to provide search and rescue teams with accurate information. Traffic maps such as those made by Waze and Google, which depend on continuous user input and regular flow of traffic, [stop working during disasters](https://www.citylab.com/transportation/2018/09/after-the-storm-a-flood-of-data/570640/), when few people are driving or leaving their home.

Social media offers an opportunity to improve these live traffic maps, but it also presents significant challenges. Using machine learning we have identified and mapped road closures posted about on [Twitter](https://twitter.com) and integrated them with live traffic reports from [HERE.com](https://www.here.com/). This information can be used by emergency responders during disasters to improve their response time and save more lives.

Our final product is a [Tableau map](https://public.tableau.com/profile/arielle7797#!/vizhome/EmergencyRoadClosures/RoadClosures?publish=yes) that can be repopulated with live traffic data whenever the data collection notebook is run. Currently, the map is populated with data in Raleigh, NC from Thursday, April 25.

This repository includes:
- A notebook that can be run for live data collection
- Jupyter notebooks walking through our project development process
- Code for potential next steps using Google Maps, Bokeh, and Flask
- Recommendations for future iterations of the project

### Contents
- Notebook pre-requisites
- Historical data collection and cleaning
- Model development and analysis
- Live data collection
- Mapping Process
- Conclusions and Recommendations
- Project team

### Notebook Prerequisites:

In order to run our code, you will need to acquire API credentials from both Twitter and HERE.com

**Twitter API:**
- In order to use the Twitter API, users are required to have an account and a valid phone number.
- Register and create [Twitter Application](https://apps.twitter.com/) to get API access keys and tokens.

**HERE.com API:**
- Register for a HERE.com developer API key and tokens for free [Here](https://developer.here.com/?create=Freemium-Basic&keepState=true&step=account).

### Historcical Data Collection and Cleaning

In order to train our model, we decided to find tweets from Hurricane Florence in September 2018. The twitter API has restrictions on historical data collection:

```
The Twitter Search API is not meant to be an exhaustive source of Tweets. Not all Tweets will be indexed or made available via the search interface.
Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.
```
To overcome this, we used documentation from "Get old tweets from Twitter" (source listed below) and colleted the tweets from 01/01/2016 till 04/14/2019. In the end, we sourced historical data from 79 Traffic and News accounts in North and South Carolina, and using the Twitter API and the Tweepy tool we collected 147799 historical tweets.

Of those, 8,000 were from the time period of Hurricane Florence and it was those we used to build our model to determine what tweets were and were not related to road closures.

#### Cleaning and Exploratory Data Analysis

Of the 8000 tweets between 9/10/18-9/30/18 (time period for Hurricane Florence), we eliminated the ones where the tweet was empty, or if it was a duplicate tweet.   We also removed the irrelevant columns.

Our data was then ready for NLP.  We used Regular Expressions and Lemmatization to remove any unnecessary words or punctuations from the tweets.

Our data was unsupervised, so we needed to create a target variable - whether a tweet has road closure information or not.  We tried 2 approaches:
1. Using key words to identify which tweets had information on road closures
2. Hardcoding the road closure tweets

The model performed better with the first approach, so we used that for the final model.  We had 3 lists:
word_list1 = ['road', 'street','rd', 'hwy', 'highway', 'ave', 'avenue','intersection']
word_list2 = ['closed','closure', 'blocked', 'flooded']
not_word_list = ['lane closed', 'lane closure','cleared', 're opened', 'reopen']

For a tweet to be a road closure tweet, it needed to have at least 1 word from word_list1, at least 1 word from word_list2, and could not have a word from not_word_list.

### Model Development and Analysis

In order to build a model, we had to first tokenize the words.  We tried CountVectorizer and TFIDFVectorizer. We created a function for fitting and scoring a model, the function included a Pipeline (vectorizer, estimator) and GridSearch.  This allowed us to easily test multiple models.

Our biggest challenge was that our classes were extremely imbalanced, ~350 road closures vs 7500+ non-road closures.  So we needed to use techniques to handle imbalanced classes.

In terms of evaluating our models, we had to make sure we reduced the False Positives, i.e. model predicts a non-road closure when it is actually closed.  Because we don't want the emergency responders to get stuck.

So instead of using Accuracy score as the metric (default for GridSearch), we used ROC AUC score, so we could minimize the False Positives.

We started out with Logistic Regression, and found the optimal  parameters using TFIDFVectorizer as that worked better than CountVectorizer.

Our ROC AUC score wasn't great, so we tried Gradient Boosting.  That boosted our score, and reduced False Positives.

Finally, we used SMOTE technique to synthetically generate more samples for the minority class (road closures).  This helped the model enormously.  We were able to get a ROC AUC score of 0.96, and only 6 tweets out of 349 were wrongly predicted.

We also tried KMeans and DBSCAN, but those models didn't work very well for us.

### Live Data Collection

Our live data collection process requires the input of an address that a user wants live traffic information for. We then use Selenium, a web browser automation tool, to search for exact location coordinates that can be used by both the HERE.com and Twitter search tools as a search parameter. For Twitter, this location data was used to create a circular radius. For HERE.com, this data was used to create what map developers call a bounding box-

**Twitter Data**
The Twitter data is collected using a search function that finds tweets within a 25mi radius of the inputted address related to a list of 9 search terms.

Only tweets with location data are collected. Location data on Twitter can be in one of two formats:
1. A single set of location coordinates tagged to the tweet
    - These tweets are instantly mappable in our program.
2. A city or area tagged to the tweet--for example, Raleigh, NC--that includes bounding box data
    - These must be parsed for intersection data or other information that can then be mapped. Although past iterations of this project were able to build a model that could read a very limited, professionally written set of tweets about traffic, we were unable to improve upon that model in order to make it applicable to tweets written by the general public, and therefore did not include it here. Their documentation can be found [here](https://github.com/balak4/Optimizing-Evac-Routes). The challenges of developing a machine learning tool that can read tweets containing intersections written by humans are vast, and require significant investment and time to develop.

Once we have a list of tweets with the necessary location data, we can run them through our function, classifying them as about road closures or not.

After that, we run them through a secondary filtering tool to make sure we are only mapping tweets with full closure information. These tweets are added to our list of mappable points.

**HERE.com Data**

The HERE.com API allows you to collect data within a bounding box at different severity levels. For this tool, we only mapped "critical" level incidents, which indicate full road closures.

Those points are added to the twitter data and used by our mapping tool.


### Mapping Process

**Tools**
- Python (Numpy, Pandas, GeoPandas, Bokeh, GMapPlot)
- Tableau
- HTML

**Sources**
- Here.com API
- Google Maps API

**Overview**
After using Selenium to select the bounding box and converting the geocoordinates into a clean dataframe/csv, we are able to plot the road closures. This dataframe includes location data from here.com and twitter.

**Challenges**
- The biggest challenge here was how to incorporate geolocation data from two sources in an accurate way.
- From a technical standpoint, the biggest challenge was finding a way to connect the start and end points of the road closures in a visually appealing way.

**Final Products**
1. Tableau: https://public.tableau.com/profile/arielle7797#!/vizhome/EmergencyRoadClosures/RoadClosures?publish=yes
    - Using Tableau, we were able to create a single interation of a successful map.  
    - Tableau was the most optimal choice because of the mapbox integration. This shows the user real time traffic/congestion as the underlying map on which the road closures are plotted.
    - The Twitter and Here.com data are marked in different colors, which allows the user to differentiate between the sources.


2. Bokeh with Google Maps: linked in the repository   
    - Using an API connection to Google Maps and python Bokeh library, this version returns a dynamic map in html format.
    - The latitudes and longitudes are passed through as 2 separate series matched on index to form a single geolocation coordinate.
    - As a next step, the start and end points of the road closures should be mapped together (similar to the tableau map).
    - This example is a good foundation to continue to build upon for a future live site or app using python code.


### Conclusion & Recommendations

We were able to build a successful model and mapping process for tweets that included exact geolocation data, successfully mapping them in concurance with data from HERE.com to display road blocks in a 25-mile radius from an address.

Our model struggles with tweets that only include city-level location data. The size of our dataset is not large enough to train a robust model that easily identifies full addresses in order to map them.

1. Flask : Given that most of the code is written in Python, we recommend using flask to integrate the below in order to maintain the code in a single place and holistic manner.
    A) API Connections
    B) Twitter & Here.com Scraping and EDA
    C) Modeling
    D) Bokeh Mapping

2. Google Maps : Explore paid opportunities with Google Maps for true optimization & directions of routes that include the new mapped road closures, in addition to showing real time traffic & congestion. We could also enhance the utility by including search capabilities for first responders to type in an address/destination that requires aid.

3. Twitter : Work with Twitter to build a stronger crowdsourced dataset during emergencies

4. A More Robust Data Set : More data collected from natural disasters on traffic to help our model successfully read tweets


### Sources
* **Documentation**

[Twitter Developer](https://developer.twitter.com/en/docs.html)

[Tweepy API](http://docs.tweepy.org/en/v3.5.0/index.html)

[Get old tweets from Twitter](https://pypi.org/project/GetOldTweets3/)


### Project Team:
* **Arielle Miro**
* **Aruna Rayapeddi**
* **Deepa Mahidhara**
* **Julia Neumann**
