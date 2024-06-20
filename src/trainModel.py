#
# importing libraries and function
#
import numpy as np
import pandas as pd  #pasdas is used to create/structure data frame
from sklearn.model_selection import train_test_split  #we need to split our data set into training data and testing data
from sklearn.feature_extraction.text import TfidfVectorizer  #convert text to data( mail data to numerical value)
from sklearn.linear_model import LogisticRegression  #if we don't import LogisticRegression, we need to built from scratch( train data)
from sklearn.metrics import accuracy_score  # split data into train, test so we need to use this function to evaluate the accuracy
import matplotlib.pyplot as plt
import seaborn as sns
#
# data collection
#

# loading the data from csv file
raw_train_data = pd.read_csv('../mail_data.csv')
sns.countplot(x='Category', data=raw_train_data)
plt.show()

# replace null values with null string
# explain code: for each element in raw_train_data, if the element is not null (True), it keeps its original value.
# if the element is null (False), it replaces it with an empty string ''
train_data = raw_train_data.where((pd.notnull(raw_train_data)), '')

# check the size of the data set( rows & columns)
# print(f"Size of the trainning data set {train_data.shape}")

# encode label to numerical ham:1, spam: 0
    # train_data['Category'] access 'Category' column
    # train_data['Category'] == 'spam' returns boolean value
    # loc() :
        # is used to access a group of rows and columns by labels or boolean arrays
        # gets, or sets, the value(s) of the specified labels
        # dataframe.loc[row, column]
train_data.loc[train_data['Category'] == 'spam', 'Category'] = 0
train_data.loc[train_data['Category'] == 'ham', 'Category'] = 1

# split train data into feature( Message) and target( Category)
# Message feature as input X
X = train_data['Message']
# print(X)
# Category as output Y
Y = train_data['Category']
# print(Y)

# split data( X, Y) into train data & test data
    # take 20%( 0.2) as text data, 80%( 0.8) as training data
    # random_state( not important): when we use 'train_test_split' function, our data will be splitted in different ways
    # random_state simply sets a seed to the random generator, so that your train-test splits are always deterministic.
    # If you don't set a seed, it is different each time.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# feature extraction: transform the text data( X) to feature vector( numerical( that can be used as input for Logistic
# regression model
    # feature extraction: the mapping from texture data to real valued vectors( numerical representation of this textual data)
    # feature extraction algorithm tries to create a list with all the unique words present in the particular text,
             # it removes the repeated words
    # bags of words( BOW): list of unique words in the text corpus( collection of words)
    # term frequency - inverse document frequency( TF-IDF): to count number of times each word appears in a document
        # term-frequency( TF) = number of times term t appearsin a document / number of terms in document
        # Inverse document frequency( IDF) = log(N/n)
            # N: number of documents
            # n: number of documents a term t has appeared in.
            # The IDF of rare word is high, whereas the IDF of a frequent word is low
        # TF-IDF value of a term = TF x IDF ( feature vector)
    # TfidfVectorizer: to count the numeber of times each word appears in the document:
        # we create a list of all words in the paragraph or in document
        # count the number of times the word repeat
    # min_df = 1: if the word repeated is less than 1 , we ignore it( not important -> ignore)
    # stop_words:
        # specifies the stop words to be removed from teh text before processing
        # stop_word = 'english': common English stop word( 'and', 'the', 'is', ...) will be removed
    # lowercase = 'True': all the letter will be convert to towercase
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
# fit_transform: fit all the input data into vectorizer function and transform to feature vector( numerical value)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train, Y_test to integer
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#
# training the Model
#

logistic_regression_model = LogisticRegression()

# traning the Logistic Regression model with the training data
logistic_regression_model.fit(X_train_features, Y_train)

# evaluating the trained model
# prediction on training data
predictions_on_training_data = logistic_regression_model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, predictions_on_training_data)
print(f"Accuracy on training data: {train_accuracy}")
# prediction on test data
predictions_on_testing_data = logistic_regression_model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, predictions_on_testing_data)
print(f"Accuracy on testing data: {test_accuracy}")

# plotting the accuracies
accuracies = [train_accuracy, test_accuracy]
labels = ['Training Accuracy', 'Test Accuracy']

plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, color=['orange', 'blue'])
plt.ylim(0, 1)  # setting the y-axis limit from 0 to 1
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy of Logistic Regression Model')
plt.show()
#
# building a predictive system
#
# input email
input_email = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."]
#convert text to feature vector
input_data_feature = feature_extraction.transform(input_email)
predictions_on_input_data = logistic_regression_model.predict(input_data_feature)
print(f"Spam-0, Ham-1: {predictions_on_input_data}")