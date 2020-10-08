
# coding: utf-8

# In[19]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

unsupervised_validation_data = pd.read_csv('UnsupervisedValidationResults.csv',index_col=0)
unsupervised_validation_data.columns.name = 'missing value ratio'
print(unsupervised_validation_data)

fig = plt.figure()
unsupervised_plot = unsupervised_validation_data.plot(
    title="Unsupervised validation results",
    
)


unsupervised_plot.set(xlabel='noise ratio',ylabel='average absolute error')
plt.savefig('test.png')
plt.show()
plt.close(fig)

