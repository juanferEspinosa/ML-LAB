#!/usr/bin/env python
# coding: utf-8

# <br><br><center><h1 style="font-size:2.7em;color:#000">MACHINE LEARNING LAB - TUTORIAL 9</h1></center>
# <br>
# <h1 style="font-size:2em;color:#000">Juan Fernando Espinosa</h1>
# <h1 style="font-size:2em;color:#000">303158</h1>
# 
# ---
# <h3 style="font-size:2em;color:#ff4411">1. SPAM FILTER USING SVM</h3>

# <h4 style="font-size:2em;color:#2467C0">A. Build a spam filter using a pre-processed dataset</h4>
# <br>

# In[1]:


from libsvm.python.svmutil import *
import pandas as pd
import pandas as DataFrame
import numpy as np
from nltk import stem
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# In[2]:


missing_values = ['-','na','Nan','nan','n/a','?']      
column_names =['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56', '57']
D3 = pd.read_csv("spambase.data", sep=',', na_values = missing_values, names = column_names)
D3.head()


# ### Training and Testing Splitting

# In[3]:


D3_train = D3.sample(frac=0.7, random_state=1)
D3_test = D3.drop(D3_train.index)

Y_train = D3_train['57'].values
print('y_train set:',len(Y_train))
X_train = D3_train.drop(['57'], axis=1).values
print('X_train set:',len(X_train))
Y_test = D3_test['57'].values
print('y_test set:',len(Y_test))
X_test = D3_test.drop(['57'], axis=1).values
print('X_test set:',len(X_test))


# ### Mixing column name and value into Dictionary - Training

# In[4]:


x = []
for i in range(len(X_train)):
    dictionary = {}
    for j in range(len(X_train[i])):
        dictionary[j]=X_train[i][j]
    x.append(dictionary)
print('First row printing:','\n','\n', x[0:1])


# ### Mixing column name and value into Dictionary - Testing

# In[5]:


x_test = []
for i in range(len(X_test)):
    dictionary = {}
    for j in range(len(X_test[i])):
        dictionary[j]=X_test[i][j]
    x_test.append(dictionary)
print('First row printing:','\n','\n', x_test[0:1])


# ### Training and testing of the model - Classification task

# In[6]:


ranges = np.arange(0.1,100,1)
len(ranges)
accuracy_Array = []
mse_array = []
prob  = svm_problem(Y_train, x, isKernel=True)
for i in ranges:
    param = svm_parameter('-c '+str(i))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test, x_test, m)
    ACC, MSE, SCC = evaluations(Y_test, p_label)
    print('MSE = ', MSE)
    accuracy_Array.append(ACC)
    mse_array.append(MSE)


# ### Plot Accuracy and MSE outputs

# In[7]:


plt.figure(figsize=(10,6))
plt.plot(ranges, accuracy_Array, 'bs-')
plt.title('Accuracy of the model')
plt.xlabel('C values')
plt.ylabel('Accuracy')
plt.show

plt.figure(figsize=(10,6))
plt.plot(ranges, mse_array, 'go-')
plt.title('Loss of the model')
plt.xlabel('C values')
plt.ylabel('Loss values')
plt.show


# In[8]:


target_names = ['ham', 'spam']
print(classification_report(Y_test, p_label, target_names=target_names))


# <h4 style="font-size:2em;color:#2467C0">B. Pre-processed a dataset and learn SVM</h4>
# <br>

# In[9]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
print(messages.head())


# In[10]:


messages.groupby('label').describe()


# In[11]:


messages['length'] = messages['message'].map(lambda text: len(text))
print(messages.head())


# In[12]:


# Lower case for all the words
messages['message'] = messages['message'].map(lambda x: x.lower())
messages['message'].head()


# In[13]:


# Erasing punctuation of the strings
messages["message"] = messages['message'].str.replace('[^\w\s]','')
messages["message"].head()


# ### Streamline workload with pipeline

# In[14]:


pipeline = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('SVM',SVC())
])


# ### Slit dataset into train and test

# In[15]:


# Splitting into Train and Test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages['message'], messages['label'], test_size = 0.3, random_state = 1)
#X_train, X_test, y_train, y_test = train_test_split(messages['message'], messages['label'], test_size = 0.1, random_state = 1)
print("Original set", messages.shape[0], "observations")
print ("Training set", X_train.shape[0], "observations")
print ("Testing set", X_test.shape[0], "observations")


# ### Parameters to run into the model

# In[16]:


ranges = np.arange(0.1,20,1)
parameters = {'SVM__kernel':['linear', 'rbf'],'SVM__C':[0.0001,0.001,0.1,10,50,100], 'SVM__gamma':[0.1,0.01, 0.001]}

print('SVM__kernel | CV: [linear, rbf]','\n', 'C parameters | CV: [0.0001,0.001,0.1,1,10,50,100]','\n','Gama parameters | CV: [0.1,0.01, 00.1]' )


# ### Cross Validation over parameters previously stated

# In[17]:


grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)


# In[18]:


# Fit of the trainin
grid.fit(X_train, y_train)
y_hat =grid.fit(X_train, y_train)
print("Best parameter | Training (CV score=%0.3f):" % grid.best_score_)
print(grid.best_params_)
print("Test score = %3.2f" %(grid.score(X_test,y_test)))


# ### Plot of the relationship between parameters

# In[19]:


def plot_results(index='SVM__C', columns='SVM__gamma'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + index
    columns = 'param_' + columns

    df = pd.DataFrame(grid.cv_results_)

    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == grid.best_params_[col[6:]]]

    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Grid-search results')
    plt.ylabel('Gamma')
    plt.legend(title=table_mean.columns.names)
    plt.show()


plot_results(index='SVM__C', columns='SVM__gamma')

# SOURCE = []


# ### Run the training and test with the optimal values

# In[20]:


parameters = {'SVM__kernel':['rbf'],'SVM__C':[100], 'SVM__gamma':[0.01]}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)

y_true, y_pred = y_test, grid.predict(X_test)
print(classification_report(y_true, y_pred))
p_label, p_acc, p_val = svm_predict(Y_test, x_test, m)
ACC, MSE, SCC = evaluations(Y_test, p_label)
cm = confusion_matrix(y_test, y_pred)
df = pd.DataFrame(cm)
df = df.rename(columns={0: 'Ham', 1: 'Spam'}, index={0: 'Predicted Ham', 1: 'Predicted Spam'})
df


# In[21]:


plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# <h3 style="font-size:2em;color:#ff4411">2. Compare SVM based spam filter with another model</h3>

# ### Streamline workload with pipeline

# In[22]:


logreg=LogisticRegression()
pipeline_lg = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf',logreg)
])


# ### Parameters to run into the model

# In[23]:


clf__C=[0.0001,0.001,0.1,10,50,100]
parameter_lg = {'clf__C':[0.0001,0.001,0.1,10,50,100]}


# ### Cross Validation over parameters previously stated

# In[24]:


grid_lg = GridSearchCV(pipeline_lg, param_grid=parameter_lg, cv=5)


# In[25]:


# Fit of the trainin
grid_lg.fit(X_train, y_train)
print("Best parameter | Training (CV score=%0.3f):" % grid_lg.best_score_)
print(grid_lg.best_params_)
print("Test score = %3.2f" %(grid_lg.score(X_test,y_test)))


# ### Run the training and test with the optimal values

# In[26]:


parameters1 = {'clf__C':[100]}
grid_lg1 = GridSearchCV(pipeline_lg, param_grid=parameters1, cv=5)
grid_lg1.fit(X_train, y_train)
y_true, y_pred = y_test, grid_lg1.predict(X_test)
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_test, y_pred)
df = pd.DataFrame(cm)
df = df.rename(columns={0: 'Ham', 1: 'Spam'}, index={0: 'Predicted Ham', 1: 'Predicted Spam'})
df


# In[27]:


plt.matshow(cm,fignum= 1)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# <h4 style="font-size:2em;color:#2467C0">Conclusions</h4>
# <br>

#  <ol>
#      <li>The test score does not change in the comparison between the SVM and Logistic models.</li><br>
#      <li>Regarding to the accuracy, there is no big difference on the outputs considering the confusion matrix and table the SVM has a lightly improvement against the Logistic Model.</li><br>
#     <li>The dataset structure may be influencing the similarity between the models. Therefore, it is not possible to make conclusions based on the information found. </li> <br>
#     <li>The distribution in train and test dataset influences as well the accuracy for each model.</li><br>
#      <li>The kernel used in SVM plays a major role: it transforms the data for solving complex problems.</li><br>
# 
#       
#   

# <h3 style="font-size:2em;color:#ff4411">3. Bibliography</h3>

# > https://www.kaggle.com/darshnaik/sms-spam-or-ham
# 
# > https://etav.github.io/projects/spam_message_classifier_naive_bayes.html
# 
# > https://radimrehurek.com/data_science_python/
# 
# > https://www.kaggle.com/darshnaik/sms-spam-or-ham
# 
# > https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
# 
# > https://pactools.github.io/auto_examples/plot_grid_search.html
# 
# > https://blancas.io/sklearn-evaluation/api/plot.html
