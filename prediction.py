
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
import itertools
import numpy as np
import seaborn as sb
import pickle

df = pd.read_csv("news.csv")
df.shape
df.head


def create_distribution(dataFile):
    return sb.countplot(x='label', data=dataFile, palette='hls')


# by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(df)

# ## Integrity Check(missing value)


def data_qualityCheck():
    print("Checking data qualitites...")
    df.isnull().sum()
    df.info()
    print("check finished.")


data_qualityCheck()
# the dataset does not contains missing values therefore no cleaning required

# Separate the label

# Separate the labels and set up training and test datasets
y = df.label
y.head()

# ## Now we can drop the label column
df.drop("label", axis=1)
# Split dataset for training and test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y, test_size=0.33, random_state=53)

X_train.head(10)
X_test.head(10)

# Extracting Of Features
# Before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
# of unstructured data into some uniform set of attributes that an algorithm can understand.
# creating feature vector - document term matrix
# Initialize the 'count_vectorizer'
count_vectorizer = CountVectorizer(stop_words='english')

# ## Fit and transform the training data
# It will Learn the vocabulary dictionary and return term-document matrix
count_train = count_vectorizer.fit_transform(X_train)
print(count_vectorizer)
print(count_train)
# Printing the doc term matrix


def get_countVectorizer_stats():
    print(count_train.shape)  # vocab size
    print(count_vectorizer.vocabulary_)  # check vocabulary using below command


get_countVectorizer_stats()

count_test = count_vectorizer.transform(X_test)

# Transforming test set
# Create tf-df frequency feature
# tf-idf
# Initialize a TfidfVectorizer
# Initialize the 'tfidf_vectorizer'
# This removes words which appear in more than 70% of the articles as given

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set and transform test set
# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)


def get_tfidf_stats():
    tfidf_train.shape
    # get train data feature names
    print(tfidf_train.A[:10])


get_tfidf_stats()

# Transform the test set

tfidf_test = tfidf_vectorizer.transform(X_test)
# Get feature names
# Get the feature names of 'tfidf_vectorizer'

print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])
count_df = pd.DataFrame(
    count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(
    tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)
# Check whether the DataFrames are equal

print(count_df.equals(tfidf_df))
print(count_df.head())
print(tfidf_df.head())
# Function to plot the confusion matrix
# This function prints and plots the confusion matrix
# Normalization can be applied by setting 'normalize=True'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Building classifier using naive bayes
nb_pipeline = Pipeline([
    ('NBTV', tfidf_vectorizer),
    ('nb_clf', MultinomialNB())])

# Fit Naive Bayes classifier according to X, y

nb_pipeline.fit(X_train, y_train)

# Perform classification on an array of test vectors X
predicted_nbt = nb_pipeline.predict(X_test)

score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score*100,2)}%')
cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)
nbc_pipeline = Pipeline([
    ('NBCV', count_vectorizer),
    ('nb_clf', MultinomialNB())])
nbc_pipeline.fit(X_train, y_train)
predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print(f'Accuracy: {round(score*100,2)}%')
cm1 = metrics.confusion_matrix(y_test, predicted_nbc, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])
print(cm1)


print(metrics.classification_report(y_test, predicted_nbt))

print(metrics.classification_report(y_test, predicted_nbc))
# Building Passive Aggressive Classifier
# Initialize a PassiveAggressiveClassifier
linear_clf = Pipeline([
    ('linear', tfidf_vectorizer),
    ('pa_clf', PassiveAggressiveClassifier(max_iter=50))])
linear_clf.fit(X_train, y_train)

# Predict on the test set and calculate accuracy

pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print(f'Accuracy: {round(score*100,2)}%')
# Build confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

print(cm)
print(metrics.classification_report(y_test, pred))
# Saving best model to the disk
# Pickle is used to dump model in file
model_file = 'finalmodel.sav'
pickle.dump(linear_clf, open(model_file, 'wb'))

var = input("Enter the news text you want to verify: ")

# function to run for prediction


def detecting_fake_news(var):

    load_model = pickle.load(open('finalmodel.sav', 'rb'))
    prediction = load_model.predict([var])

    return (print("The given statement is ", prediction[0]))


if __name__ == '__main__':
    detecting_fake_news(var)
