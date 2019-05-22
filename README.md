
# NLP MODELING FOR AIRCRAFT MAINTENANCE 

_By Deepa Mahidhara_

Aircrafts have maintenance issues that are identified and written up as maintenance logs during various stages of the operations.These open logs need to be addressed by the maintenance operations team to keep the aircraft in service.
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



**CONTENTS**

- Presentation: NLP Modeling pdf file (in "Presentation" folder)

- Data (in "data" folder)

    Historical data provided (Input):
      Ds&Cs2015.xlsx - maintenance and delays data for 2015
      2016Ds&Cs.xlsx - maintenance and delays data for 2016

    Intermediate data:
      ata_all_codes.csv - ATA chapters info from Wikipedia
      ata_25_codes.csv - from airline
      data_ata_all.csv - all Observations
      data_ata25.csv - Observations for ATA Chapter 25
      processed_data.csv - data ready for NLP
      vectorized_data_tvec.csv - tokens for modeling

    Output data:
      matching_logs.csv - for any log, outputs log with highest similarity, and corresponding action

- Notebooks (in "Notebooks" folder)

  1. capstone1_data_cleaning.ipynb - Cleans input data
  Input:
    Ds&Cs2015.xlsx
    2016Ds&Cs.xlsx
  Output:
    data_ata_all.csv
    data_ata25.csv

  2. capstone2_EDA.ipynb - EDA
  Input:
    data_ata25.csv
  Output:
    None

  3. capstone3_preprocessing.ipynb - Preprocessing - RegEx, Lemmatize, etc.
  Input:
    data_ata25.csv
  Output:
    processed_data.csv

  4. capstone4_Vectorizer.ipynb - Generating tokens
  Input:
    data_ata25.csv
  Output:
    vectorized_data_tvec.csv

  5. capstone5_LogReg.ipynb - Logistic Regression model
  Input:
    vectorized_data_tvec.csv
  Output:
    None

  6. capstone6_FFNN.ipynb - Feed Forward Neural Network model
  Input:
    vectorized_data_tvec.csv
  Output:
    None

  7. capstone7_cosine_similarity.ipynb - Calculating cosine similarity
  Input:
    processed_data.csv
  Output:
    None - Feed Forward Neural Network model
  Input:
    vectorized_data_tvec.csv
  Output:
    matching_logs.csv

  8. capstone8_spacy.ipynb - Using Spacy to dentify Nouns and Verbs
  Input:
    processed_data.csv
  Output:
    None - Feed Forward Neural Network model
  Input:
    processed_data.csv
  Output:
    None

**PROCESS**

### Historical Data Collection and Cleaning

I have Maintenance data from an airline for 2015 and 2016; this also includes data on Delays and Cancellations.

No of variables:
No of Observations for 2017:
No of Observations for 2017:  

I went through this data, changed data type where necessary (eg converted ATA codes to object and added leading zeros where needed), identified Null values, dropped duplicates.

I also got ATA codes and descriptions from the Web, and cleaned up where necessary.

The Airline gave me their classification/ descriptions for ATA Chapter 25.

#### Exploratory Data Analysis

I then performed EDA (Exploratory Data Analysis) to see what the key words were, average length of documents for problem log and corrective action, and distribution of classes.

I looked at Words of 1, 2 and 3 lengths.

This helped me decide what I needed to do in the Preprocessing step.

### Preprocessing

Now I had to get the ready for NLP.  This included:
Changing all text to lower case
Lemmatizing
Lots of Reg Ex to prepare the data

### Vectorizer

Next step was to generate tokens for text data.  I tried CountVectorizer, TFIDFVectorizer and Word2Vec, and found that TFIDFVectorizer worked the best.

I created a function for fitting and scoring a model, the function included a Pipeline (vectorizer, estimator) and GridSearch.  This allowed me to easily test multiple models.

### Model Development and Analysis

Since predicting action was difficult, I built a model to predict ATA codes.

My biggest challenge was that my classes were extremely imbalanced.  Out of 7 classes, 85% of my data was in 1 class.

I tried 3 models:

Logistic Regression: My Train score was 1, but Test score was .7, clearly my model had severe overfitting.

SMOTE and Gradient Boost: I then applied SMOTE to synthetically generate more samples for the minority classes.  This helped me reduced overfitting.

Feed Forward Neural Network:  Finally I used FFNN, my Train and Test scores were both ~0,85, so clearly my model was not overfit.

### Using Cosine Similarity

My next task was to predict action from historical logs and ATA codes.

I found that most frequently used words in Problem Log were Nouns, whereas most frequently used words in Corrective Action were Verbs.

My approach was as follows:
  For a new log,
    i. Find the historical log with highest Cosine Similarity
    ii. Find corresponding corrective action that was taken
    iii. Take noun from Problem Log and verb from Corrective Action
    iv. Concatenate Nouns and Verbs to predict action - have not done this yet


### Using Spacy to identify Nouns and Verbs

I used Spacy to identify Nouns and Verbs.  Spacy also identified Pronouns and Adjectives that will be useful in predicting Action.


**Tools**
- Python (Numpy, Pandas, Scikit Learn, Matplotlib, FuzzyWuzzy, Spacy)

**Sources**
- Wikipedia.com

**Challenges**
- The biggest challenge here was how to handle Tech data as there is no Corpus readily available.
- Dealing with imbalanced classes was another challenge.
- From a technical standpoint, the biggest challenge was to build a model for a combination of text and numerical data.


### EXECUTIVE SUMMARY

It is very difficult to predict Action as Tech terms are used in the logs.
The logs are short, but the action phrases are much longer.
Often, it is not easy to see the similarity between Logs and Actions.

I was able to build a successful model to predict ATA classes for ATA Chapter 25 from historical log and action data.  
SMOTE and Gradient Boosting helped me reduced overfitting from Logistic Regression.
Feed Forward Neural Networks worked even better, but I didn't have time to try CNN and RNN.

I started the process of predicting Action from Logs, however, for my model to be successful:
  1. I need to preprocess the data further (more RegEx to get rid of irrelevant numerical data)
  2. Find a corpus of Tech words - specifically for airline maintenance
  3. Learn Spacy and use Topic Modeling to identify similar contextual phrases.
  4. Try Unsupervised Learning.
