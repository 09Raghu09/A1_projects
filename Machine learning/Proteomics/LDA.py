# Linear Discrimant Analysis
# # Week 10 - Group N

import numpy as np
import pandas as pd
import itertools
from tabulate import tabulate

import progressbar
import random

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

print('All packages successfully loaded')

#frame = pd.read_excel("stage2_woman_men_feature_tidy.csv")
frame = pd.read_csv("stage2_woman_men_feature_tidy.csv", sep=",")
X = np.array(frame["intensity"])
Y = np.array(frame["class"])

# frame_2 = pd.read_excel("output_tidyf.xlsx")

# X_2 = np.array(frame_2["intensity"])# 
# Y_2 = np.array(frame_2["class"])

# X = np.array(data["intensity"])
# Y = np.array(data["class"])

# X = np.concatenate((X,X_2))
# Y = np.concatenate((Y,Y_2))

random_state=85
# Split Data into Train (2/3) and Test (1/3)
DataTrain, DataTest, YTrain, YTest = train_test_split(X, Y, test_size=1/3, stratify=Y, random_state=random_state)
print("Training and testing data successfully split with test size = 1/3")

# Create the model 
model = LinearDiscriminantAnalysis()

# Fit on training data
model.fit(DataTrain, YTrain)

print("Training of the model is done.")


# ## 6. Test & Validate ML Model

# In[66]:


# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(DataTrain)
train_rf_probs = model.predict_proba(DataTrain)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(DataTest)
rf_probs = model.predict_proba(DataTest)[:, 1]

print('Training and testing data used to make predictions, to demonstrate overfitting and to determine performance')


# ## 7. Evaluate ML Model
# In order to evaluate the ML model, we need to implement an evaluate method. This way we can generate a ROC curve, with the AUC values, a confusion matrix and more analysis which might be needed.

# ### ROC Curve

# In[67]:


def evaluate_model(predictions, probs, train_predictions, train_probs, train_labels, test_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['accuracy'] = accuracy_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['auc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['accuracy'] =  accuracy_score(test_labels, predictions)
    results['auc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['accuracy'] =  accuracy_score(train_labels, train_predictions)
    train_results['auc'] = roc_auc_score(train_labels, train_probs)
    
    
    metrics = []
    for metric in ['recall', 'precision', 'accuracy', 'auc']:
        metrics.append((metric,round(baseline[metric], 4), round(train_results[metric], 4), round(results[metric], 4)))
    print(tabulate(metrics, headers=['Metric', 'Baseline', 'Train', 'Test']))
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    testing_fpr, testing_tpr, _ = roc_curve(test_labels, probs)
    training_fpr, training_tpr, _ = roc_curve(train_labels, train_probs)
    
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    baseline_label = 'Baseline - AUC: %s' %(round(baseline['auc'], 4))
    testing_label = 'Testing - AUC: %s' %(round(results['auc'], 4))
    training_label = 'Training - AUC: %s' %(round(train_results['auc'], 4))
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'grey', label = baseline_label)
    plt.plot(testing_fpr, testing_tpr, 'y', label = testing_label)
    plt.plot(training_fpr, training_tpr, 'g', label = training_label)
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curve');
    plt.show();


# In[68]:


evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, YTrain, YTest)
plt.savefig('results/roc_auc_curve.png')


# ### Confusion Matrix

# In[69]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# Generate a confusion matrix for the **training** data. 

# In[70]:


training_cm = confusion_matrix(YTrain, train_rf_predictions)
plot_confusion_matrix(training_cm, classes = ['Healthy', 'Cancer'],
                      title = 'Training Confusion Matrix')

plt.savefig('results/training_cm.png')


# Generate a confusion matrix for the **testing** data.

# In[71]:


cm = confusion_matrix(YTest, rf_predictions)
plot_confusion_matrix(cm, classes = ['Healthy', 'Cancer'],
                      title = 'Testing Confusion Matrix')

plt.savefig('results/testing_cm.png')


# ### Feature importance analysis

# In[72]:


# Extract feature importances
fi = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_})


# #### Extract the top 18 features 

# In[73]:


sorted_features = fi.sort_values(by='importance', ascending=False)
important_features = sorted_features[:18]


# In[74]:


important_features.to_csv("results/important_features.csv", sep='\t')


# Plot the top 18 features 

# In[75]:


import plotly_express as px

px.bar(important_features, x='feature', y='importance', labels={'feature':'Features', 'importance':'Feature importance'})


# Save the plot in results folder

# In[76]:


fig = px.bar(important_features, x='feature', y='importance', labels={'feature':'Features', 'importance':'Feature importance'})
fig.write_image("results/important_features.png")


# ### Permutation Importance

# In[ ]:


def PermImportance(X, y, clf, metric, num_iterations=100):
    '''
    Calculates the permutation importance of features in a dataset.
    Inputs:
    X: dataframe with all the features
    y: array-like sequence of labels
    clf: sklearn classifier, already trained on training data
    metric: sklearn metric, such as accuracy_score, precision_score or recall_score
    num_iterations: no. of repetitive runs of the permutation
    Outputs:
    baseline: the baseline metric without any of the columns permutated
    scores: differences in baseline metric caused by permutation of each feature, dict in the format {feature:[diffs]}
    '''
    max_length = len(X.columns)
    bar=progressbar.ProgressBar(max_length)
    bar.start()
    baseline_metric=metric(y, clf.predict(X))
    scores={c:[] for c in X.columns}
    for c in X.columns:
        X1=X.copy(deep=True)
        for _ in range(num_iterations):
            temp=X1[c].tolist()
            random.shuffle(temp)
            X1[c]=temp
            score=metric(y, clf.predict(X1))
            scores[c].append(baseline_metric-score)
        
        bar.update(X.columns.tolist().index(c))
    bar.finish()
    return baseline_metric, scores


# In[ ]:


baseline, scores=PermImportance(DataTest, YTest, model, accuracy_score, num_iterations=10)


# In[ ]:


percent_changes={c:[] for c in X.columns}
for c in scores:
    for i in range(len(scores[c])):
        percent_changes[c].append(scores[c][i]/baseline*100)


# In[ ]:


px.bar(
    pd.DataFrame.from_dict(percent_changes).melt().groupby(['variable']).mean().reset_index().sort_values(['value'], ascending=False)[:25], 
    x='variable', 
    y='value', 
    labels={
        'variable':'column', 
        'value':'% change in recall'
        }
       )


# In[ ]:





# In[ ]:




