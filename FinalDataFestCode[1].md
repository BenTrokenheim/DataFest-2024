```python
pip install statsmodels
```

    Requirement already satisfied: statsmodels in c:\users\btrok\anaconda3\envs\example\lib\site-packages (0.14.2)
    Requirement already satisfied: numpy>=1.22.3 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from statsmodels) (1.26.4)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in c:\users\btrok\anaconda3\envs\example\lib\site-packages (from statsmodels) (1.13.0)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from statsmodels) (2.2.1)
    Requirement already satisfied: patsy>=0.5.6 in c:\users\btrok\anaconda3\envs\example\lib\site-packages (from statsmodels) (0.5.6)
    Requirement already satisfied: packaging>=21.3 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from statsmodels) (23.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2024.1)
    Requirement already satisfied: six in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from patsy>=0.5.6->statsmodels) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install networkx
```

    Requirement already satisfied: networkx in c:\users\btrok\anaconda3\envs\example\lib\site-packages (3.3)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\users\btrok\anaconda3\envs\example\lib\site-packages (1.4.2)
    Requirement already satisfied: numpy>=1.19.5 in c:\users\btrok\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in c:\users\btrok\anaconda3\envs\example\lib\site-packages (from scikit-learn) (1.13.0)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\btrok\anaconda3\envs\example\lib\site-packages (from scikit-learn) (1.4.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\btrok\anaconda3\envs\example\lib\site-packages (from scikit-learn) (3.4.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx
```


```python
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
```


```python
# Define the path to the files
base_path = ''

# Load each CSV file into a DataFrame
checkpoints_df = pd.read_csv('checkpoints.csv')
items_df = pd.read_csv('items.csv')
media_views_df = pd.read_csv('media_views.csv')
page_views_df = pd.read_csv('page_views.csv')
responses_df = pd.read_csv('responses.csv')
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\67440926.py:8: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
      page_views_df = pd.read_csv('page_views.csv')
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\67440926.py:9: DtypeWarning: Columns (32,33,34,35,36,37,38,39) have mixed types. Specify dtype option on import or set low_memory=False.
      responses_df = pd.read_csv('responses.csv')
    

Difficulty Across Chapters


```python
columns_to_keep = ['book', 'release', 'chapter', 'page', 'chapter_number',
                   'section_number', 'review_flag', 'institution_id', 'class_id',
                   'student_id', 'item_id', 'item_type', 'response', 'prompt',
                   'points_possible', 'points_earned', 'dt_submitted', 'completes_page',
                   'attempt']

responses_df = responses_df.loc[:, columns_to_keep]

responses_df['accuracy'] = responses_df['points_earned'] / responses_df['points_possible']
```


```python
# Calculate mean accuracy, standard deviation, and sample size for each chapter
chapter_stats = responses_df.groupby('chapter')['accuracy'].agg(['mean', 'std', 'count']).reset_index()

# Extract chapter numbers and handle special chapters
chapter_stats['chapter_num'] = chapter_stats['chapter'].str.extract('(\d+)').astype(float)  # Extract numeric part
# chapter_stats.loc[chapter_stats['chapter'].str.contains("Getting Started", na=False), 'chapter_num'] = 0  # Assign 0 to "Getting Started"
# chapter_stats.loc[chapter_stats['chapter'] == "Practice Exam", 'chapter_num'] = chapter_stats['chapter_num'].max() + 1  # Assign max+1 to "Practice Exam"

# Sort by this custom order
chapter_stats_sorted = chapter_stats.sort_values(by='chapter_num').dropna()

# Calculate the SEM (Standard Error of the Mean)
chapter_stats_sorted['SEM'] = chapter_stats_sorted['std'] / np.sqrt(chapter_stats_sorted['count'])

# Plotting
plt.figure(figsize=(18, 18))
plt.bar(chapter_stats_sorted['chapter'], chapter_stats_sorted['mean'], yerr=chapter_stats_sorted['SEM'] * 2, color='skyblue', capsize=5)
plt.xlabel('Chapter')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy by Chapter with SEM Error Bars')

# Customizing x-axis labels to display only "Getting Started", numeric chapters, and "Practice Exam"
xticks_labels = ['Getting Started' if x == 0 else f'Chapter {int(x)}' if pd.notnull(x) else 'Practice Exam' for x in chapter_stats_sorted['chapter_num']]
plt.xticks(ticks=np.arange(len(xticks_labels)), labels=xticks_labels, rotation=90)

plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    



```python
# Plotting with improvements
plt.figure(figsize=(18, 10))  # Adjusted for better fit

# Set the background color
plt.gca().set_facecolor('#f0f0f0')  # Light grey background for contrast
plt.grid(color='white', linestyle='--', linewidth=0.5)  # Adding gridlines for readability

# Adjusting the color of the bars and making error bars red
bar_color = '#102747'  # Dark blue shade
error_bar_props = {'ecolor': 'red', 'capsize': 5, 'capthick': 2, 'elinewidth': 2}  # Customizing error bars to be red and more visible

plt.bar(chapter_stats_sorted['chapter'], chapter_stats_sorted['mean'], yerr=chapter_stats_sorted['SEM'] * 3,
        color=bar_color, edgecolor='grey', error_kw=error_bar_props)

# Customizing the Y-axis range
# Adjust these limits based on your data's specific range
plt.ylim([min(chapter_stats_sorted['mean'] - chapter_stats_sorted['SEM'] * 3) * 0.9, 
          max(chapter_stats_sorted['mean'] + chapter_stats_sorted['SEM'] * 3) * 1.1])

plt.xlabel('Chapter', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Mean Accuracy by Chapter with SEM Error Bars', fontsize=16)

# Customizing x-axis labels to display only "Getting Started", numeric chapters, and "Practice Exam"
xticks_labels = ['Getting Started' if x == 0 else f'Chapter {int(x)}' if pd.notnull(x) else 'Practice Exam' for x in chapter_stats_sorted['chapter_num']]
plt.xticks(ticks=np.arange(len(xticks_labels)), labels=xticks_labels, rotation=45, ha='right')  # Adjusted rotation for readability

plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    



```python
chapter_stats_sorted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapter</th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>chapter_num</th>
      <th>SEM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chapter 1 - Welcome to Statistics: A Modeling ...</td>
      <td>0.771259</td>
      <td>0.420025</td>
      <td>66241</td>
      <td>1.0</td>
      <td>0.001632</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Midterm 1</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Chapter 2 - Understanding Data</td>
      <td>0.696690</td>
      <td>0.459690</td>
      <td>121450</td>
      <td>2.0</td>
      <td>0.001319</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Chapter 3 - Examining Distributions</td>
      <td>0.656664</td>
      <td>0.474824</td>
      <td>136997</td>
      <td>3.0</td>
      <td>0.001283</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Chapter 4 - Explaining Variation</td>
      <td>0.640985</td>
      <td>0.479713</td>
      <td>141802</td>
      <td>4.0</td>
      <td>0.001274</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Chapter 5 - A Simple Model</td>
      <td>0.665731</td>
      <td>0.471737</td>
      <td>75906</td>
      <td>5.0</td>
      <td>0.001712</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Chapter 6 - Quantifying Error</td>
      <td>0.626610</td>
      <td>0.483706</td>
      <td>128434</td>
      <td>6.0</td>
      <td>0.001350</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Chapter 7 - Adding an Explanatory Variable to ...</td>
      <td>0.609800</td>
      <td>0.487798</td>
      <td>84959</td>
      <td>7.0</td>
      <td>0.001674</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Chapter 8 - Digging Deeper into Group Models</td>
      <td>0.622576</td>
      <td>0.484745</td>
      <td>97299</td>
      <td>8.0</td>
      <td>0.001554</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Chapter 9 - Models with a Quantitative Explana...</td>
      <td>0.607129</td>
      <td>0.488391</td>
      <td>120948</td>
      <td>9.0</td>
      <td>0.001404</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chapter 10 - The Logic of Inference</td>
      <td>0.577078</td>
      <td>0.494027</td>
      <td>74924</td>
      <td>10.0</td>
      <td>0.001805</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chapter 11 - Model Comparison with F</td>
      <td>0.537262</td>
      <td>0.498613</td>
      <td>76901</td>
      <td>11.0</td>
      <td>0.001798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chapter 12 - Parameter Estimation and Confiden...</td>
      <td>0.575820</td>
      <td>0.494221</td>
      <td>79603</td>
      <td>12.0</td>
      <td>0.001752</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chapter 13 - Introduction to Multivariate Models</td>
      <td>0.602655</td>
      <td>0.489371</td>
      <td>10847</td>
      <td>13.0</td>
      <td>0.004699</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chapter 13 - What You Have Learned</td>
      <td>0.548163</td>
      <td>0.497710</td>
      <td>7049</td>
      <td>13.0</td>
      <td>0.005928</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chapter 14 - Multivariate Model Comparisons</td>
      <td>0.503922</td>
      <td>0.500009</td>
      <td>10198</td>
      <td>14.0</td>
      <td>0.004951</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Chapter 15 - Models with Interactions</td>
      <td>0.508675</td>
      <td>0.499960</td>
      <td>7032</td>
      <td>15.0</td>
      <td>0.005962</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Chapter 16 - More Models with Interactions</td>
      <td>0.455132</td>
      <td>0.498033</td>
      <td>4959</td>
      <td>16.0</td>
      <td>0.007072</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Assuming 'responses_df' is your DataFrame
# Group by 'student_id' and 'chapter' to calculate mean accuracy
mean_accuracy = responses_df.groupby(['student_id', 'chapter'])['accuracy'].mean().reset_index(name='mean_accuracy')

# Extract chapter number for each row in 'mean_accuracy'
mean_accuracy['chapter_num'] = mean_accuracy['chapter'].str.extract('(\d+)').astype(float)
```


```python
# Ensure no NaN values for the regression analysis
mean_accuracy.dropna(subset=['mean_accuracy', 'chapter_num'], inplace=True)

# Define predictor (X) and response (Y) variables
X = mean_accuracy[['chapter_num']]  # Predictor
X = sm.add_constant(X)  # Adds a constant term to the predictor
Y = mean_accuracy['mean_accuracy']  # Response

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>mean_accuracy</td>  <th>  R-squared:         </th> <td>   0.106</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.106</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1934.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 27 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>18:29:31</td>     <th>  Log-Likelihood:    </th> <td>  3949.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16298</td>      <th>  AIC:               </th> <td>  -7895.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16296</td>      <th>  BIC:               </th> <td>  -7879.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    0.7482</td> <td>    0.003</td> <td>  247.178</td> <td> 0.000</td> <td>    0.742</td> <td>    0.754</td>
</tr>
<tr>
  <th>chapter_num</th> <td>   -0.0174</td> <td>    0.000</td> <td>  -43.979</td> <td> 0.000</td> <td>   -0.018</td> <td>   -0.017</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>659.506</td> <th>  Durbin-Watson:     </th> <td>   0.750</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 697.433</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-0.484</td>  <th>  Prob(JB):          </th> <td>3.58e-152</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.698</td>  <th>  Cond. No.          </th> <td>    15.8</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Plotting the linear regression results
plt.figure(figsize=(18, 6))
plt.scatter(mean_accuracy['chapter_num'], mean_accuracy['mean_accuracy'], color='blue', label='Data Points')
plt.plot(mean_accuracy['chapter_num'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('Chapter Number')
plt.ylabel('Mean Accuracy')
plt.title('Linear Regression of Mean Accuracy on Chapter Number')
plt.legend()
plt.show()
```


    
![png](output_13_0.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming `mean_accuracy`, `model`, and `X` are defined as per your context

# Your code for model preparation and prediction goes here...

# Plotting the linear regression results with enhanced styling
plt.figure(figsize=(18, 6))  # Adjusted for a wider plot

# Set the background color and add gridlines for better readability
plt.gca().set_facecolor('#f0f0f0')  # Light grey background for contrast
plt.grid(color='white', linestyle='--', linewidth=0.5)

# Scatter plot for data points
plt.scatter(mean_accuracy['chapter_num'], mean_accuracy['mean_accuracy'], color='#102747', label='Data Points')  # Dark blue for data points

# Regression line
plt.plot(mean_accuracy['chapter_num'], model.predict(X), color='red', label='Regression Line')  # Red for the regression line

# Customizing the plot with labels and title
plt.xlabel('Chapter Number', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Linear Regression of Mean Accuracy on Chapter Number', fontsize=16)

# Adding a legend
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](output_14_0.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Sort the DataFrame by chapter number
mean_accuracy_sorted = mean_accuracy.sort_values(by='chapter_num')

# Plot setup
plt.figure(figsize=(18, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(mean_accuracy_sorted['chapter_num'].unique())))

# Plot each chapter's KDE
for i, chapter in enumerate(mean_accuracy_sorted['chapter_num'].unique()):
    data = mean_accuracy_sorted[mean_accuracy_sorted['chapter_num'] == chapter]['mean_accuracy']
    kde = gaussian_kde(data)
    y = np.linspace(0, 1, 1000)
    x = kde(y)
    
    # Offset for vertical separation
    x_offset = i # Adjust the multiplier for desired separation
    
    plt.fill_betweenx(y, x_offset, x + x_offset, alpha=0.5, color=colors[i])
    plt.plot(x + x_offset, y, color=colors[i], label=f'Chapter {chapter}')

plt.xlabel('Chapter', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Mean Accuracy by Chapter with SEM Error Bars', fontsize=16)

# Customizing x-axis labels to display only "Getting Started", numeric chapters, and "Practice Exam"
xticks_labels = ['Getting Started' if x == 0 else f'Chapter {int(x)}' if pd.notnull(x) else 'Practice Exam' for x in chapter_stats_sorted['chapter_num']]
plt.xticks(ticks=np.arange(len(xticks_labels)), labels=xticks_labels, rotation=45, ha='right')  # Adjusted rotation for readability

plt.title('Separated KDE of Mean Accuracy for Each Chapter')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Calculate residuals
residuals = model.resid

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Q-Q plot of the residuals
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
sm.qqplot(residuals, line='s', ax=ax)  # 's' indicates a standardized line
plt.title('Q-Q Plot of Residuals')
plt.show()
```


    
![png](output_16_0.png)
    



    
![png](output_16_1.png)
    


Relationship Between Section Responses and Review Responses


```python
responses_df = pd.read_csv(f'{base_path}responses.csv')

columns_to_keep = ['book', 'release', 'chapter', 'page', 'chapter_number',
                   'section_number', 'review_flag', 'institution_id', 'class_id',
                   'student_id', 'item_id', 'item_type', 'response', 'prompt',
                   'points_possible', 'points_earned', 'dt_submitted', 'completes_page',
                   'attempt']

responses_df = responses_df.loc[:, columns_to_keep]

responses_df['accuracy'] = responses_df['points_earned'] / responses_df['points_possible']
responses_df.dropna(subset=['accuracy'], inplace=True)

mean_accuracy = responses_df.groupby(['student_id', 'chapter'])['accuracy'].mean().reset_index(name='mean_accuracy')
responses_df = pd.merge(responses_df, mean_accuracy, on=['student_id', 'chapter'], how='left')
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\3018772879.py:1: DtypeWarning: Columns (32,33,34,35,36,37,38,39) have mixed types. Specify dtype option on import or set low_memory=False.
      responses_df = pd.read_csv(f'{base_path}responses.csv')
    


```python
# Step 1: Subset checkpoints_df to keep only specific columns
checkpoints_subset = checkpoints_df[['student_id', 'chapter_number', 'response', 'construct', 'EOC']]
# Step 2: Merge the DataFrames
merged_df = pd.merge(responses_df, checkpoints_subset, on=['student_id', 'chapter_number'], how='left')
```


```python
# Scatter plot of mean_accuracy vs EOC
plt.figure(figsize=(18, 6))
sns.scatterplot(data=merged_df, x='EOC', y='mean_accuracy')
plt.title('Relationship between Mean Accuracy and EOC')
plt.xlabel('EOC')
plt.ylabel('Mean Accuracy')
plt.show()
```


    
![png](output_20_0.png)
    



```python
# Step 1: Identify chapters as review or non-review
# This step assumes you have a 'chapter' column to work with
merged_df['is_review'] = merged_df['page'].str.contains("Review")

# Step 2: Calculate mean accuracies separately for review and non-review
# For Review Chapters
review_mean_accuracy = merged_df[merged_df['is_review']].groupby(['student_id', 'chapter_number'])['accuracy'].mean().reset_index(name='review_mean_accuracy')

# For Non-Review Chapters
non_review_mean_accuracy = merged_df[~merged_df['is_review']].groupby(['student_id', 'chapter_number'])['accuracy'].mean().reset_index(name='non_review_mean_accuracy')

# Step 3: Merge these calculations back into merged_df
# Since we're adding two new columns, we'll perform two merge operations

# Merge review mean accuracy
merged_df = pd.merge(merged_df, review_mean_accuracy, on=['student_id', 'chapter_number'], how='left')

# Merge non-review mean accuracy
merged_df = pd.merge(merged_df, non_review_mean_accuracy, on=['student_id', 'chapter_number'], how='left')
```


```python
# Ensure no NaN values for the regression analysis
data_for_regression = merged_df.dropna(subset=['review_mean_accuracy', 'non_review_mean_accuracy', ])

# Define predictor (X) and response (Y) variables
X = data_for_regression[['non_review_mean_accuracy']]  # Predictor
X = sm.add_constant(X)  # Adds a constant term to the predictor
Y = data_for_regression['review_mean_accuracy']  # Response

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>review_mean_accuracy</td> <th>  R-squared:         </th>  <td>   0.379</td> 
</tr>
<tr>
  <th>Model:</th>                     <td>OLS</td>         <th>  Adj. R-squared:    </th>  <td>   0.379</td> 
</tr>
<tr>
  <th>Method:</th>               <td>Least Squares</td>    <th>  F-statistic:       </th>  <td>2.197e+06</td>
</tr>
<tr>
  <th>Date:</th>               <td>Sat, 27 Apr 2024</td>   <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                   <td>18:31:20</td>       <th>  Log-Likelihood:    </th> <td>1.1785e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>        <td>3600215</td>       <th>  AIC:               </th> <td>-2.357e+06</td>
</tr>
<tr>
  <th>Df Residuals:</th>            <td>3600213</td>       <th>  BIC:               </th> <td>-2.357e+06</td>
</tr>
<tr>
  <th>Df Model:</th>                <td>     1</td>        <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>nonrobust</td>      <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                    <td>    0.2020</td> <td>    0.000</td> <td>  601.366</td> <td> 0.000</td> <td>    0.201</td> <td>    0.203</td>
</tr>
<tr>
  <th>non_review_mean_accuracy</th> <td>    0.7509</td> <td>    0.001</td> <td> 1482.390</td> <td> 0.000</td> <td>    0.750</td> <td>    0.752</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>41216.713</td> <th>  Durbin-Watson:     </th> <td>   0.389</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>73205.523</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.007</td>   <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 3.698</td>   <th>  Cond. No.          </th> <td>    7.81</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Scatter plot of non-review accuracy vs. review accuracy
plt.figure(figsize=(18, 18))
sns.scatterplot(x='non_review_mean_accuracy', y='review_mean_accuracy', data=merged_df, color='blue', label='Data Points')

# Calculate predictions for the regression line
X_plot = sm.add_constant(merged_df['non_review_mean_accuracy'])
Y_pred = model.predict(X_plot)

# Plot the regression line
plt.plot(merged_df['non_review_mean_accuracy'], Y_pred, color='red', label='Regression Line')

plt.title('Non-Review Accuracy vs. Review Accuracy')
plt.xlabel('Non-Review Mean Accuracy')
plt.ylabel('Review Mean Accuracy')
plt.legend()
plt.show()
```


    
![png](output_23_0.png)
    



```python
# Drop any rows that could not be converted or were missing
merged_df.dropna(subset=['review_mean_accuracy', 'non_review_mean_accuracy', 'chapter_number'], inplace=True)

# Define the model formula for the mixed-effects model
# Here, 'chapter_number' is treated as a numeric fixed effect
model_formula = 'review_mean_accuracy ~ non_review_mean_accuracy'

# Fit the mixed-effects model with 'student_id' as the random effect
mixed_model = smf.mixedlm(model_formula, merged_df, groups=merged_df['student_id'])
mixed_model_result = mixed_model.fit()

mixed_model_result.summary()
```




<table class="simpletable">
<tr>
       <td>Model:</td>       <td>MixedLM</td> <td>Dependent Variable:</td> <td>review_mean_accuracy</td>
</tr>
<tr>
  <td>No. Observations:</td> <td>3600215</td>       <td>Method:</td>               <td>REML</td>        
</tr>
<tr>
     <td>No. Groups:</td>     <td>1472</td>         <td>Scale:</td>               <td>0.0176</td>       
</tr>
<tr>
  <td>Min. group size:</td>     <td>3</td>      <td>Log-Likelihood:</td>       <td>2160722.2490</td>    
</tr>
<tr>
  <td>Max. group size:</td>   <td>24888</td>      <td>Converged:</td>               <td>Yes</td>        
</tr>
<tr>
  <td>Mean group size:</td>  <td>2445.8</td>           <td></td>                     <td></td>          
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>             <th>Coef.</th> <th>Std.Err.</th>    <th>z</th>    <th>P>|z|</th> <th>[0.025</th> <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>                <td>0.378</td>   <td>0.004</td>  <td>102.416</td> <td>0.000</td>  <td>0.371</td>  <td>0.385</td>
</tr>
<tr>
  <th>non_review_mean_accuracy</th> <td>0.463</td>   <td>0.001</td>  <td>612.155</td> <td>0.000</td>  <td>0.462</td>  <td>0.465</td>
</tr>
<tr>
  <th>Group Var</th>                <td>0.020</td>   <td>0.005</td>     <td></td>       <td></td>       <td></td>       <td></td>   
</tr>
</table><br/>





```python
# Assuming 'merged_df' is sorted by 'chapter_number'
merged_df = merged_df.sort_values(by='chapter_number')

# Define the unique chapter numbers
unique_chapters = merged_df['chapter_number'].unique()

# Settings for the plot
plt.figure(figsize=(20, 20))  # Adjust the figure size as necessary

# Number of chapters to plot
num_chapters_to_plot = 8  # Targeting n plots

for i, chapter in enumerate(unique_chapters[:num_chapters_to_plot], 1):
    plt.subplot(4, 4, i)  # 4x4 grid for 16 plots
    chapter_data = merged_df[merged_df['chapter_number'] == chapter]
    
    # Scatter plot for this chapter
    sns.scatterplot(x='non_review_mean_accuracy', y='review_mean_accuracy', data=chapter_data, color='blue', label=f'Chapter {chapter} Data Points')
    
    # Assuming you have a regression model for each chapter or a global model applied here
    # For demonstration, let's assume a global model and we're plotting the line based on available chapter data
    # Calculate predictions for the regression line specific to this chapter
    if not chapter_data.empty:
        X_plot = sm.add_constant(chapter_data['non_review_mean_accuracy'], has_constant='add')
        try:
            Y_pred = model.predict(X_plot)
            # Plot the regression line
            plt.plot(chapter_data['non_review_mean_accuracy'], Y_pred, color='red', label='Regression Line')
        except ValueError as e:
            print(f"Skipping chapter {chapter} due to error: {e}")
    
    plt.title(f'Ch. {chapter}: Non-Review vs. Review Accuracy')
    plt.xlabel('Non-Review Accuracy')
    plt.ylabel('Review Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\833541905.py:37: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      plt.tight_layout()
    C:\Users\btrok\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\IPython\core\pylabtools.py:152: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)
    


    
![png](output_25_1.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd

# Assuming 'merged_df' and 'model' are defined as per your context

# Sorting by 'chapter_number'
merged_df = merged_df.sort_values(by='chapter_number')

# Define the unique chapter numbers
unique_chapters = merged_df['chapter_number'].unique()

# Settings for the plot
plt.figure(figsize=(20, 10))  # Adjust the figure size for clarity and layout

# Number of chapters to plot
num_chapters_to_plot = 8  # Targeting n plots

for i, chapter in enumerate(unique_chapters[:num_chapters_to_plot], 1):
    plt.subplot(2, 4, i)  # Adjusted grid for better visibility
    chapter_data = merged_df[merged_df['chapter_number'] == chapter]
    
    # Scatter plot for this chapter
    sns.scatterplot(x='non_review_mean_accuracy', y='review_mean_accuracy', data=chapter_data, color='#102747', alpha=0.7, label=f'Ch. {chapter} Data Points')
    
    # Plotting the regression line
    if not chapter_data.empty:
        X_plot = sm.add_constant(chapter_data['non_review_mean_accuracy'], has_constant='add')
        try:
            Y_pred = model.predict(X_plot)
            plt.plot(chapter_data['non_review_mean_accuracy'], Y_pred, color='red', linewidth=2, label='Regression Line')
        except ValueError as e:
            print(f"Skipping chapter {chapter} due to error: {e}")
    
    plt.title(f'Ch. {chapter}: Non-Review vs. Review Accuracy', fontsize=10)
    plt.xlabel('Non-Review Accuracy', fontsize=9)
    plt.ylabel('Review Accuracy', fontsize=9)
    plt.legend(prop={'size': 8})

plt.tight_layout()
plt.show()
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\3574769902.py:41: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      plt.tight_layout()
    C:\Users\btrok\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\IPython\core\pylabtools.py:152: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)
    


    
![png](output_26_1.png)
    



```python
# Initialize lists to store results
chapter_numbers = []
intercepts = []
coefficients = []

# Loop through each unique chapter to perform regression
for chapter in merged_df['chapter_number'].unique():
    # Filter data for the current chapter
    chapter_data = merged_df[merged_df['chapter_number'] == chapter]
    
    # Check if there's enough data to perform regression
    if len(chapter_data) > 1:
        # Define predictor and response variables
        X = chapter_data[['non_review_mean_accuracy']]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        Y = chapter_data['review_mean_accuracy']
        
        # Fit the linear regression model
        model = sm.OLS(Y, X).fit()
        
        # Store the results
        chapter_numbers.append(chapter)
        intercepts.append(model.params[0])
        coefficients.append(model.params[1])

# Convert the lists to a DataFrame for easier plotting
regression_params_df = pd.DataFrame({
    'Chapter': chapter_numbers,
    'Intercept': intercepts,
    'Coefficient': coefficients
})
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:23: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      intercepts.append(model.params[0])
    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\4237604450.py:24: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      coefficients.append(model.params[1])
    


```python
# Plotting the intercepts and coefficients over chapters
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(regression_params_df['Chapter'], regression_params_df['Intercept'], marker='o', linestyle='-')
plt.title('Intercept Change Over Chapters')
plt.xlabel('Chapter Number')
plt.ylabel('Intercept')

plt.subplot(1, 2, 2)
plt.plot(regression_params_df['Chapter'], regression_params_df['Coefficient'], marker='o', linestyle='-')
plt.title('Coefficient Change Over Chapters')
plt.xlabel('Chapter Number')
plt.ylabel('Coefficient')

plt.tight_layout()
plt.show()
```


    
![png](output_28_0.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming `regression_params_df` is your DataFrame with 'Chapter', 'Intercept', and 'Coefficient' columns

plt.figure(figsize=(14, 6))

# Setting the overall background color
plt.gcf().set_facecolor('#f0f0f0')

# Intercept Scatter Plot with Regression Line
plt.subplot(1, 2, 1)
plt.scatter(regression_params_df['Chapter'], regression_params_df['Intercept'], color='#102747')  # Dark blue for scatter
# Calculate and plot regression line for intercepts
z = np.polyfit(regression_params_df['Chapter'], regression_params_df['Intercept'], 1)
p = np.poly1d(z)
plt.plot(regression_params_df['Chapter'], p(regression_params_df['Chapter']), "r--")  # Red dashed regression line
plt.title('Intercept Change Over Chapters')
plt.xlabel('Chapter Number')
plt.ylabel('Intercept')
plt.grid(color='white', linestyle='--', linewidth=0.5)  # Adding gridlines for readability

# Coefficient Scatter Plot with Regression Line
plt.subplot(1, 2, 2)
plt.scatter(regression_params_df['Chapter'], regression_params_df['Coefficient'], color='#102747')  # Dark blue for scatter
# Calculate and plot regression line for coefficients
z = np.polyfit(regression_params_df['Chapter'], regression_params_df['Coefficient'], 1)
p = np.poly1d(z)
plt.plot(regression_params_df['Chapter'], p(regression_params_df['Chapter']), "r--")  # Red dashed regression line
plt.title('Coefficient Change Over Chapters')
plt.xlabel('Chapter Number')
plt.ylabel('Coefficient')
plt.grid(color='white', linestyle='--', linewidth=0.5)  # Adding gridlines for readability

plt.tight_layout()
plt.show()
```


    
![png](output_29_0.png)
    



```python
# Regression for Intercept Change Over Chapters
X_intercept = sm.add_constant(regression_params_df['Chapter'])  # Adds a constant term
Y_intercept = regression_params_df['Intercept']
model_intercept = sm.OLS(Y_intercept, X_intercept).fit()

# Regression for Coefficient Change Over Chapters
X_coefficient = sm.add_constant(regression_params_df['Chapter'])  # Adds a constant term
Y_coefficient = regression_params_df['Coefficient']
model_coefficient = sm.OLS(Y_coefficient, X_coefficient).fit()
```


```python
# Display the regression summary for intercepts
print("Regression Statistics for Intercept Change Over Chapters:")
model_intercept.summary()
```

    Regression Statistics for Intercept Change Over Chapters:
    

    C:\Users\btrok\anaconda3\envs\example\Lib\site-packages\scipy\stats\_axis_nan_policy.py:531: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=9
      res = hypotest_fun_out(*samples, **kwds)
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Intercept</td>    <th>  R-squared:         </th> <td>   0.717</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.677</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   17.74</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 27 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>0.00398</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:38:38</td>     <th>  Log-Likelihood:    </th> <td>  10.502</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>     9</td>      <th>  AIC:               </th> <td>  -17.00</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>     7</td>      <th>  BIC:               </th> <td>  -16.61</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    0.4682</td> <td>    0.062</td> <td>    7.545</td> <td> 0.000</td> <td>    0.322</td> <td>    0.615</td>
</tr>
<tr>
  <th>Chapter</th> <td>   -0.0465</td> <td>    0.011</td> <td>   -4.212</td> <td> 0.004</td> <td>   -0.073</td> <td>   -0.020</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.747</td> <th>  Durbin-Watson:     </th> <td>   1.805</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.688</td> <th>  Jarque-Bera (JB):  </th> <td>   0.597</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.266</td> <th>  Prob(JB):          </th> <td>   0.742</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.856</td> <th>  Cond. No.          </th> <td>    12.6</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Display the regression summary for coefficients
print("\nRegression Statistics for Coefficient Change Over Chapters:")
model_coefficient.summary()
```

    
    Regression Statistics for Coefficient Change Over Chapters:
    

    C:\Users\btrok\anaconda3\envs\example\Lib\site-packages\scipy\stats\_axis_nan_policy.py:531: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=9
      res = hypotest_fun_out(*samples, **kwds)
    




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Coefficient</td>   <th>  R-squared:         </th> <td>   0.486</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.413</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   6.623</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 27 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>0.0368</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>18:38:39</td>     <th>  Log-Likelihood:    </th> <td>  7.0754</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>     9</td>      <th>  AIC:               </th> <td>  -10.15</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>     7</td>      <th>  BIC:               </th> <td>  -9.756</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    0.4958</td> <td>    0.091</td> <td>    5.460</td> <td> 0.001</td> <td>    0.281</td> <td>    0.711</td>
</tr>
<tr>
  <th>Chapter</th> <td>    0.0415</td> <td>    0.016</td> <td>    2.573</td> <td> 0.037</td> <td>    0.003</td> <td>    0.080</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.402</td> <th>  Durbin-Watson:     </th> <td>   1.421</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.818</td> <th>  Jarque-Bera (JB):  </th> <td>   0.064</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.140</td> <th>  Prob(JB):          </th> <td>   0.969</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.698</td> <th>  Cond. No.          </th> <td>    12.6</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Prompt Analysis


```python
responses_df = pd.read_csv(f'{base_path}responses.csv')

columns_to_keep = ['book', 'release', 'chapter', 'page', 'chapter_number',
                   'section_number', 'review_flag', 'institution_id', 'class_id',
                   'student_id', 'item_id', 'item_type', 'response', 'prompt',
                   'points_possible', 'points_earned', 'dt_submitted', 'completes_page',
                   'attempt']

responses_df = responses_df.loc[:, columns_to_keep]

responses_df['accuracy'] = responses_df['points_earned'] / responses_df['points_possible']
responses_df.dropna(subset=['accuracy'], inplace=True)

mean_accuracy = responses_df.groupby(['student_id', 'chapter'])['accuracy'].mean().reset_index(name='mean_accuracy')
responses_df = pd.merge(responses_df, mean_accuracy, on=['student_id', 'chapter'], how='left')
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\3018772879.py:1: DtypeWarning: Columns (32,33,34,35,36,37,38,39) have mixed types. Specify dtype option on import or set low_memory=False.
      responses_df = pd.read_csv(f'{base_path}responses.csv')
    


```python
# Step 1: Subset checkpoints_df to keep only specific columns
checkpoints_subset = checkpoints_df[['student_id', 'chapter_number', 'response', 'construct', 'EOC']]
# Step 2: Merge the DataFrames
merged_df = pd.merge(responses_df, checkpoints_subset, on=['student_id', 'chapter_number'], how='left')
```


```python
from collections import Counter

# Splitting strings and extracting keywords
keywords = " ".join(merged_df['prompt'].astype(str).unique()).split()
keyword_counts = Counter(keywords)

# Selecting the top N keywords to display in the bar graph
top_n = 50
top_keywords = keyword_counts.most_common(top_n)
words = [item[0] for item in top_keywords]
frequencies = [item[1] for item in top_keywords]

# Creating a bar graph
plt.figure(figsize=(18, 10))
plt.bar(words, frequencies, color='skyblue')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.title('Top Keywords in Prompts')
plt.xticks(rotation=45, ha="right")
plt.show()
```


    
![png](output_36_0.png)
    


Attempt and Accuracy


```python
# Assuming 'accuracy' is a binary column where 1 indicates correct response and 0 indicates incorrect
prob_accuracy_by_attempt = merged_df.groupby('attempt')['accuracy'].mean().reset_index()
prob_accuracy_by_attempt.rename(columns={'accuracy': 'probability_of_accuracy_1'}, inplace=True)
# Group by 'chapter_number' and 'attempt', then calculate the mean of 'accuracy'
prob_accuracy_by_attempt_chapter = merged_df.groupby(['chapter_number', 'attempt'])['accuracy'].mean().reset_index()
prob_accuracy_by_attempt_chapter.rename(columns={'accuracy': 'probability_of_accuracy_1'}, inplace=True)
# Calculate the overall mean accuracy for each attempt across all chapters
overall_prob_accuracy_by_attempt = merged_df.groupby('attempt')['accuracy'].mean().reset_index()
overall_prob_accuracy_by_attempt.rename(columns={'accuracy': 'overall_probability_of_accuracy_1'}, inplace=True)

plt.figure(figsize=(16, 8))

# Plot individual chapters' accuracy probability as before
sns.lineplot(data=prob_accuracy_by_attempt_chapter, x='attempt', y='probability_of_accuracy_1', hue='chapter_number', marker='o', palette='tab10', legend='full')

# Overlay the aggregated accuracy probability with a thicker line
sns.lineplot(data=overall_prob_accuracy_by_attempt, x='attempt', y='overall_probability_of_accuracy_1', color='black', linewidth=2.5, label='Aggregated', marker='o')

plt.title('Probability of Accuracy = 1 by Attempt Number Across Chapters')
plt.xlabel('Attempt Number')
plt.ylabel('Probability of Accuracy = 1')
plt.legend(title='Chapter Number', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_38_0.png)
    



```python
percentiles = [0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]  # Define the percentiles you're interested in
attempt_percentiles = merged_df['attempt'].quantile(percentiles).to_dict()

# Print the percentiles
for percentile, value in attempt_percentiles.items():
    print(f"{percentile * 100}th percentile: {value}")
```

    89.0th percentile: 1.0
    90.0th percentile: 1.0
    91.0th percentile: 2.0
    92.0th percentile: 2.0
    93.0th percentile: 2.0
    94.0th percentile: 2.0
    95.0th percentile: 2.0
    96.0th percentile: 3.0
    97.0th percentile: 3.0
    98.0th percentile: 4.0
    99.0th percentile: 7.0
    


```python
# Filter for attempts up to and including the 10th
filtered_prob_accuracy_by_attempt_chapter = prob_accuracy_by_attempt_chapter[prob_accuracy_by_attempt_chapter['attempt'] <= 10]
filtered_overall_prob_accuracy_by_attempt = overall_prob_accuracy_by_attempt[overall_prob_accuracy_by_attempt['attempt'] <= 10]

plt.figure(figsize=(14, 8))

# Plot individual chapters' accuracy probability up to the 10th attempt
sns.lineplot(data=filtered_prob_accuracy_by_attempt_chapter, x='attempt', y='probability_of_accuracy_1', hue='chapter_number', marker='o', palette='tab10', legend='full', alpha = 0.1)

# Overlay the aggregated accuracy probability up to the 10th attempt with a thicker line
sns.lineplot(data=filtered_overall_prob_accuracy_by_attempt, x='attempt', y='overall_probability_of_accuracy_1', color='black', linewidth=2.5, label='Aggregated', marker='o')

plt.title('Probability of Accuracy = 1 by Attempt Number (Up to 10th Attempt) Across Chapters')
plt.xlabel('Attempt Number')
plt.ylabel('Probability of Accuracy = 1')
plt.legend(title='Chapter Number', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_40_0.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'filtered_prob_accuracy_by_attempt_chapter' and 'filtered_overall_prob_accuracy_by_attempt' are predefined

plt.figure(figsize=(14, 8))

# Setting background and gridlines
plt.gcf().set_facecolor('#f0f0f0')
plt.grid(color='white', linestyle='--', linewidth=0.5)

# Plot individual chapters' accuracy probability up to the 10th attempt
sns.lineplot(data=filtered_prob_accuracy_by_attempt_chapter, x='attempt', y='probability_of_accuracy_1', hue='chapter_number', 
             marker='o', palette='tab10', legend='full', alpha=0.15)  # Adjusted alpha for better visibility

# Overlay the aggregated accuracy probability up to the 10th attempt with a thicker line
sns.lineplot(data=filtered_overall_prob_accuracy_by_attempt, x='attempt', y='overall_probability_of_accuracy_1', color='#102747', 
             linewidth=2.5, label='Aggregated', marker='o')  # Dark blue color for aggregated line

plt.title('Probability of Accuracy = 1 by Attempt Number (Up to 10th Attempt) Across Chapters', fontsize=16)
plt.xlabel('Attempt Number', fontsize=14)
plt.ylabel('Probability of Accuracy = 1', fontsize=14)

# Legend settings
plt.legend(title='Chapter Number', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, framealpha=1, edgecolor='#f0f0f0')

# Adjusting the legend's background to match the figure's background
plt.legend(title='Chapter Number', bbox_to_anchor=(1.05, 1), loc='upper left')
frame = legend.get_frame()
frame.set_color('#f0f0f0')

plt.tight_layout()
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[141], line 29
         27 # Adjusting the legend's background to match the figure's background
         28 plt.legend(title='Chapter Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    ---> 29 frame = legend.get_frame()
         30 frame.set_color('#f0f0f0')
         32 plt.tight_layout()
    

    NameError: name 'legend' is not defined



    
![png](output_41_1.png)
    



```python
# Formula: 'accuracy ~ attempt' specifies that 'accuracy' is predicted by 'attempt'
# 'groups=df_for_model['student_id']' treats 'student_id' as a random effect
model_formula = 'accuracy ~ attempt'

# Fit the mixed-effects model
mixed_model = smf.mixedlm(model_formula, merged_df, groups=merged_df['student_id'])
mixed_model_result = mixed_model.fit()

# Print the summary of the model results
mixed_model_result.summary()
```

    C:\Users\btrok\anaconda3\envs\example\Lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    C:\Users\btrok\anaconda3\envs\example\Lib\site-packages\statsmodels\regression\mixed_linear_model.py:2200: ConvergenceWarning: Retrying MixedLM optimization with lbfgs
      warnings.warn(
    




<table class="simpletable">
<tr>
       <td>Model:</td>       <td>MixedLM</td> <td>Dependent Variable:</td>   <td>accuracy</td>   
</tr>
<tr>
  <td>No. Observations:</td> <td>4930340</td>       <td>Method:</td>           <td>REML</td>     
</tr>
<tr>
     <td>No. Groups:</td>     <td>1588</td>         <td>Scale:</td>           <td>0.2080</td>    
</tr>
<tr>
  <td>Min. group size:</td>     <td>2</td>      <td>Log-Likelihood:</td>   <td>-3128624.0852</td>
</tr>
<tr>
  <td>Max. group size:</td>   <td>33122</td>      <td>Converged:</td>           <td>Yes</td>     
</tr>
<tr>
  <td>Mean group size:</td>  <td>3104.7</td>           <td></td>                 <td></td>       
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>       <th>Coef.</th> <th>Std.Err.</th>     <th>z</th>    <th>P>|z|</th> <th>[0.025</th> <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>  <td>0.662</td>   <td>0.003</td>   <td>197.079</td> <td>0.000</td>  <td>0.656</td>  <td>0.669</td>
</tr>
<tr>
  <th>attempt</th>   <td>-0.035</td>   <td>0.000</td>  <td>-253.350</td> <td>0.000</td> <td>-0.035</td> <td>-0.034</td>
</tr>
<tr>
  <th>Group Var</th>  <td>0.017</td>   <td>0.001</td>      <td></td>       <td></td>       <td></td>       <td></td>   
</tr>
</table><br/>




Behavior and Performance


```python
# First, aggregate merged_df to ensure unique combinations of student_id and chapter_number
aggregated_df = merged_df.groupby(['student_id', 'chapter_number'])['mean_accuracy'].mean().reset_index()
# Now, merge the aggregated data with page_views_df
page_views_df = pd.merge(page_views_df, aggregated_df, on=['student_id', 'chapter_number'], how='left')
```


```python
# Assuming page_views_df is your DataFrame and dt_accessed is in a recognizable datetime string format
page_views_df['dt_accessed'] = pd.to_datetime(page_views_df['dt_accessed'])
# Now that dt_accessed is a datetime object, you can sort the DataFrame by this column
page_views_df = page_views_df.sort_values(by='dt_accessed')
```


```python
# Combine chapter_number and section_number into a unique identifier for each access event.
page_views_df['chapter_section'] = page_views_df['chapter_number'].astype(str) + '_' + page_views_df['section_number'].astype(str)
# Group by student_id and chapter_section, then count occurrences
access_counts = page_views_df.groupby(['student_id', 'chapter_section']).size().reset_index(name='counts')
# Calculate total accesses for each student
total_accesses = access_counts.groupby('student_id')['counts'].sum().reset_index(name='total_counts')
# Merge back to get total counts per student_id-chapter_section pair
access_prob = access_counts.merge(total_accesses, on='student_id')
# Calculate probability
access_prob['probability'] = access_prob['counts'] / access_prob['total_counts']
# Calculate entropy contribution for each row
access_prob['entropy_contribution'] = -access_prob['probability'] * np.log2(access_prob['probability'])
# Sum entropy contributions by student_id to get total entropy
student_entropy = access_prob.groupby('student_id')['entropy_contribution'].sum().reset_index(name='entropy')
# Merge entropy values back into the original DataFrame
page_views_df = page_views_df.merge(student_entropy, on='student_id', how='left')
```


```python
# Count occurrences of each chapter-section combination
chapter_section_distribution = page_views_df.groupby(['chapter_number', 'section_number']).size().reset_index(name='access_count')

# Sort the distribution by access_count to see the most and least accessed chapter-sections
chapter_section_distribution = chapter_section_distribution.sort_values(by='access_count', ascending=False)

# Plotting
plt.figure(figsize=(18, 18))
plt.bar(x=chapter_section_distribution.index, height=chapter_section_distribution['access_count'])
plt.title('Distribution of Chapter-Section Accesses')
plt.xlabel('Chapter-Section Index')
plt.ylabel('Access Count')
plt.xticks(ticks=chapter_section_distribution.index, labels=[f"{row['chapter_number']}-{row['section_number']}" for idx, row in chapter_section_distribution.iterrows()], rotation=90)
plt.tight_layout()
plt.show()
```


    
![png](output_47_0.png)
    



```python
# Assuming page_views_df is your DataFrame and it contains 'student_id', 'chapter_section', and 'dt_accessed'

# Step 1: Sort the DataFrame by 'student_id' and 'dt_accessed'
page_views_df_sorted = page_views_df.sort_values(by=['student_id', 'dt_accessed'])

# Helper function to calculate conditional probabilities and conditional entropy
def calculate_conditional_entropy(access_sequence):
    # Count transitions
    transitions = pd.Series(list(zip(access_sequence, access_sequence[1:]))).value_counts()
    total_transitions = transitions.sum()
    
    # Calculate conditional probabilities P(Y|X)
    conditional_probs = transitions / total_transitions
    
    # Calculate conditional entropy H(Y|X)
    conditional_entropy = -np.sum(conditional_probs * np.log2(conditional_probs))
    
    return conditional_entropy

# Calculate joint entropy for each student
joint_entropy = {}
for student_id, group in page_views_df_sorted.groupby('student_id'):
    # Extract the chapter_section sequence for the current student
    access_sequence = group['chapter_section'].tolist()
    
    # Ensure there's at least one transition to calculate conditional entropy
    if len(access_sequence) > 1:
        joint_entropy[student_id] = calculate_conditional_entropy(access_sequence)
    else:
        joint_entropy[student_id] = 0  # Assign 0 entropy for single access, implying no uncertainty

# Convert joint entropy dictionary to a DataFrame for easy merging
joint_entropy_df = pd.DataFrame(list(joint_entropy.items()), columns=['student_id', 'joint_entropy'])

# Merge the joint entropy back to the original DataFrame (or to a summary DataFrame as needed)
page_views_df = page_views_df.merge(joint_entropy_df, on='student_id', how='left')
```


```python
# Since a student might appear multiple times with the same entropy value, we'll drop duplicates
cleaned_df = page_views_df[page_views_df['joint_entropy'] >= 5]
cleaned_df = cleaned_df.drop_duplicates(subset='student_id', keep='first')
```


```python
# Plotting the distribution of joint entropy
plt.figure(figsize=(18, 8))
plt.hist(cleaned_df['joint_entropy'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Joint Entropy per Student')
plt.xlabel('Joint Entropy')
plt.ylabel('Number of Students')
plt.grid(axis='y', alpha=0.75)
plt.show()
```


    
![png](output_50_0.png)
    



```python
# Set the aesthetic style of the plots
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# Create the plot
plt.figure(figsize=(18, 12))
ax = sns.regplot(x='joint_entropy', y='mean_accuracy', data=cleaned_df, color='#102747', 
                 line_kws={'color': 'red'}, scatter_kws={'alpha':0.35})

# Set the background color
ax.set_facecolor('#f0f0f0')

# Labeling the axes and title
plt.xlabel('Joint Entropy', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Joint Entropy vs Mean Accuracy with Linear Regression', fontsize=16)

# Adjusting limits if necessary (optional, depending on your data)
plt.xlim([cleaned_df['joint_entropy'].min() * 0.9, cleaned_df['joint_entropy'].max() * 1.1])
plt.ylim([cleaned_df['mean_accuracy'].min() * 0.9, cleaned_df['mean_accuracy'].max() * 1.1])

plt.tight_layout()
plt.show()
```


    
![png](output_51_0.png)
    



```python
# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plot KDE
plt.figure(figsize=(18, 18))
ax = sns.kdeplot(x='joint_entropy', y='mean_accuracy', data=cleaned_df, thresh=0, levels=100)

# Calculate the linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(cleaned_df['joint_entropy'], cleaned_df['mean_accuracy'])

# Plot the linear regression line
x = np.linspace(cleaned_df['joint_entropy'].min(), cleaned_df['joint_entropy'].max(), 50)
y = slope * x + intercept
# plt.plot(x, y, '-r', label=f'Linear Regression\ny={slope:.2f}x+{intercept:.2f}')

plt.title('Joint Entropy vs. Mean Accuracy with KDE', fontsize=16)
plt.xlabel('Joint Entropy', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.legend(fontsize=12)

# Setting the background color and gridlines for aesthetics matching previous enhancements
plt.gca().set_facecolor('#f0f0f0')
plt.grid(color='white', linestyle='--', linewidth=0.5)

plt.show()
```

    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
    


    
![png](output_52_1.png)
    



```python
# Drop rows where 'joint_entropy', 'mean_accuracy', or 'student_id' have NaN values
cleaned_df = cleaned_df.dropna(subset=['joint_entropy', 'mean_accuracy', 'student_id'])

# Example formula: 'mean_accuracy ~ entropy + (1|student_id)'
# This specifies 'entropy' as a fixed effect and includes a random intercept for each 'student_id'
model_formula = 'mean_accuracy ~ joint_entropy'

# Fit the mixed-effects model
md = smf.mixedlm(model_formula, cleaned_df, groups=cleaned_df['student_id'])
mdf = md.fit()

# Print the summary of the model
mdf.summary()
```

    C:\Users\btrok\anaconda3\envs\example\Lib\site-packages\statsmodels\regression\mixed_linear_model.py:2261: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
      warnings.warn(msg, ConvergenceWarning)
    




<table class="simpletable">
<tr>
       <td>Model:</td>       <td>MixedLM</td> <td>Dependent Variable:</td> <td>mean_accuracy</td>
</tr>
<tr>
  <td>No. Observations:</td>   <td>273</td>         <td>Method:</td>           <td>REML</td>     
</tr>
<tr>
     <td>No. Groups:</td>      <td>273</td>         <td>Scale:</td>           <td>0.0172</td>    
</tr>
<tr>
  <td>Min. group size:</td>     <td>1</td>      <td>Log-Likelihood:</td>      <td>67.1088</td>   
</tr>
<tr>
  <td>Max. group size:</td>     <td>1</td>        <td>Converged:</td>           <td>Yes</td>     
</tr>
<tr>
  <td>Mean group size:</td>    <td>1.0</td>            <td></td>                 <td></td>       
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>        <th>Coef.</th> <th>Std.Err.</th>   <th>z</th>   <th>P>|z|</th> <th>[0.025</th> <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>     <td>0.269</td>   <td>0.106</td>  <td>2.538</td> <td>0.011</td>  <td>0.061</td>  <td>0.477</td>
</tr>
<tr>
  <th>joint_entropy</th> <td>0.061</td>   <td>0.015</td>  <td>4.006</td> <td>0.000</td>  <td>0.031</td>  <td>0.090</td>
</tr>
<tr>
  <th>Group Var</th>     <td>0.017</td>     <td></td>       <td></td>      <td></td>       <td></td>       <td></td>   
</tr>
</table><br/>





```python
# Extract residuals
residuals = mdf.resid

from scipy.stats import norm

# Plot distribution of residuals
plt.figure(figsize=(18, 8))

# Original histogram plot
sns.histplot(residuals, bins=100, kde=False, stat="density", label="Residuals Histogram")

# Calculate mean and standard deviation for the residuals
mean, std = np.mean(residuals), np.std(residuals)

# Generate points on the x axis between the min and max of residuals
x = np.linspace(min(residuals), max(residuals), 1000)

# Calculate normal distribution with the same mean and std
normal_distribution = norm.pdf(x, mean, std)

# Plot the normal distribution
plt.plot(x, normal_distribution, label="Normal Distribution", linewidth=2, color='r')

plt.title('Distribution of Residuals with Overlayed Normal Distribution')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()

plt.show()
```


    
![png](output_54_0.png)
    



```python
import scipy.stats as stats

# Assuming 'residuals' are obtained from the fitted model as before
# Generate Q-Q plot for the residuals to check normality
plt.figure(figsize=(18, 8))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.show()
```


    
![png](output_55_0.png)
    



```python
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Assuming 'page_views_df' is your DataFrame and has been preprocessed accordingly

# Select a specific student's data
student_id = 'b9b1abc2-8934-4468-af82-68fc14534784'  # Example student ID
student_data = page_views_df[page_views_df['student_id'] == student_id].copy()
student_data = student_data.sort_values(by='dt_accessed')

# Create a Directed Graph
G = nx.DiGraph()
G.add_nodes_from(student_data['chapter_section'].unique())
for i in range(len(student_data) - 1):
    source = student_data.iloc[i]['chapter_section']
    target = student_data.iloc[i + 1]['chapter_section']
    if G.has_edge(source, target):
        G[source][target]['weight'] += 1
    else:
        G.add_edge(source, target, weight=1)

# Layout
pos = nx.kamada_kawai_layout(G)

# Nodes: Color and Size
node_color = ['gold' if node == student_data['chapter_section'].iloc[0] else 'lightgreen' for node in G]
node_size = [G.degree(node) * 500 + 500 for node in G]

# Edges: Color and Width
edge_color = ['black' if G[u][v]['weight'] > 1 else 'black' for u, v in G.edges()]
edge_width = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]

plt.figure(figsize=(24, 24))
nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=node_size, 
        edge_color=edge_color, width=edge_width, font_size=22, arrowsize=20)

plt.title(f"Enhanced Viewing Path of Student {student_id}", fontsize=20)
plt.axis('off')  # Turn off the axis
plt.show()
```


    
![png](output_56_0.png)
    



```python
# Count student appearances
student_counts = page_views_df['student_id'].value_counts()

# Sort by count (descending) and get unique student IDs
sorted_unique_students = student_counts.sort_values(ascending=True).index.tolist()

sorted_unique_students
```




    ['229f131c-fe22-40d9-a82b-1bd12898eaf3',
     'e69ab722-c400-4135-b0d6-60b90cb39e12',
     '6d566358-f02f-436c-aba1-e0afd3b5bdd7',
     '1bd66bf3-1400-41b7-856c-df718b1038dd',
     '7a26ed43-861a-41e6-8193-0bc925ab3c5a',
     'f8b9675a-12e8-4319-9c47-e4dddf547a28',
     '7a0945da-b1ca-4050-8f7f-84e58300d442',
     'af9a33a5-0c5e-4c00-a7ca-0ece7ffd6cb1',
     '41f40590-73d2-4456-a82c-af900d8f9e49',
     'ec885cdc-ca08-42b0-b9a9-550dd93d5d60',
     '77eb2c8b-cd76-4f9d-9e66-9a8662cb1753',
     '27555eb7-efcd-4684-926b-b2a049652a8c',
     '0d132aa0-0535-4c4f-b9a8-310aa61d9eda',
     '3d9df5b5-e1d7-40c5-ad24-ef6cf9e5bb62',
     'd89259e6-5688-4097-9ac7-37043297f498',
     '4f461c3f-5af5-4315-97b9-55eb2af95608',
     'c10dafe3-9350-4974-9f99-8b040ea3d47f',
     'd2131381-7142-45fc-810f-5a7219ceaa1d',
     '599c635d-b4b9-4783-a38e-39445313b903',
     '5593db75-ed48-4448-a498-abbcb527b2ee',
     '9befe8ab-6770-41f2-acd3-eacc20bf134a',
     '7740fbdb-164c-44cd-abac-04a45f49c628',
     '74549686-d400-45f6-86bd-5da0f79c2f8f',
     '046c4be5-49ef-4fe6-baa4-4062d249253a',
     '17bfb3a1-48b1-4101-8377-3290e4da385b',
     'e659a90c-eb0d-4b53-8672-40c622af6176',
     '10e4b9e9-8872-4a8f-aad2-60f49150cf71',
     'd5c3894a-e41a-42a3-b1e6-93ebbf46fdb1',
     '67d23259-7576-477d-bb39-a753887789d4',
     'd6f2bf22-a116-4212-a657-8aa70e4ad866',
     '86f5bae9-2eb7-455d-bc89-e0046e4e6119',
     '3ad4babb-2a94-41ae-8324-f57717493a23',
     '9cded0e5-da3f-4383-b755-01d94010c719',
     'b2d132a9-f051-4426-a55d-475ba62e7e78',
     '9aa26120-778c-4a2c-b006-7f2c5a485392',
     'fcc2ac8c-4937-4c5c-a236-8be1e150dcec',
     '0ae6cad4-7e39-47ee-997d-eed75684e781',
     'c308c6a7-533b-4c43-a048-26305a48aebf',
     'ea40bd31-1ab6-4671-b450-d9aaf7875b4c',
     '6b8d3a30-7ae3-439e-a181-faabe152bc35',
     'f7c9ecc6-919c-4aad-95c9-7ffb153a65cc',
     'b847ec6d-29e2-4103-92ec-69c73ea2527b',
     'ace0656f-0419-4728-8167-43d97f5d194d',
     '0ab0d3e5-2bdc-4631-94fd-78953d31c2d1',
     '41f0b6dd-48a3-4c2b-92ce-77d364f5d408',
     '2a9bddd2-b04a-4fe4-853b-49f7c39f9fec',
     'a19b960c-93e4-42ad-a369-bf444ab85432',
     'a75c4d5d-1e93-4e17-b485-841464261b98',
     '348a0c3d-6062-4cb2-a3bf-04f71cd05396',
     '61aab9d8-9c59-4aa8-ab01-c048b960fd1c',
     'ff548e42-0011-4dd7-ab46-2552503ad60c',
     '372f9f73-1634-4c2e-91e5-0200d0140995',
     'f68bf241-eb82-4674-8e6f-40535ad71d57',
     '48f140fc-2d28-4aee-9659-088199fde816',
     'f84d7ac8-c3eb-4f64-a4bc-ad45b3f8ac36',
     '052d3e01-6ab0-4d7c-8f16-093b518f1e61',
     'b8d996cb-a69d-4e7d-a385-9a6484a7d06d',
     '2e5893bc-5514-4489-9d25-9981743af538',
     'a53c10d3-a366-4c77-86fd-37991cf78408',
     '37309374-d137-4976-93a8-a2f6b2734a98',
     '1b646f9b-bf88-46f3-931c-aea2c84134e5',
     '1e0d06ef-80f8-41d0-b483-6d46a0942088',
     '4ae2aa76-96e6-412c-8451-a47be62423ed',
     '72eb6f58-68f2-43b2-be4d-8a22e8495bcc',
     'f8658da6-bf64-465f-91f6-8a3cb6128eec',
     '2946ab0a-4d28-4cbb-b421-a0df71cb4ceb',
     '4e5eb2e1-c52f-4df8-8314-cd8f6f9fb4f1',
     '8f77684b-fa34-4055-b722-6ffe965c7b92',
     '1028d510-540a-46b9-80a2-ce3b399b590b',
     '60674c5a-95e6-48f9-8883-18d2fc2017b7',
     '2d150db2-ad60-48c9-914e-31d1b9bff9ad',
     '3c729a75-e54a-4e76-bb91-f241f2bc17ed',
     '95df5ba5-c48f-4335-a5c1-0c49947e0beb',
     'd95cfa7a-63d7-4d3c-89b5-9eea3b68035a',
     'fb731e6f-3409-48e2-96ff-377e3aab423c',
     '38af660d-cea5-4df0-a669-5d9359c38aa8',
     '0e5cd67e-1632-4699-ae93-0e932ce95916',
     'f3335e26-a06e-4e22-8410-356948424c55',
     '31d04217-5986-445f-b0f3-af0edab6f965',
     '6b27d0d1-16b5-4371-9310-54e53f2c1965',
     'b63b584c-a69e-4e9d-ba57-efd9ecbeab77',
     'c48337fd-a13a-4aef-91ea-de58fca66379',
     '8f682644-bcc7-41ea-ab83-3ccebc3d2b42',
     '21b7aa18-5d8b-4c13-a18a-ecf1096290b4',
     'c971e547-2eb9-4edc-bedc-4f56ffaff828',
     '437e3420-079b-4e80-91e1-332fca34535a',
     '776f28f2-5c58-4547-98ba-81489b97a247',
     '27dfb519-b5c2-4582-9cee-8608dad5edfc',
     '86bcf9a3-2687-47e7-b422-dd8bc6d71c6b',
     '3dd4aa35-545c-4407-a09d-cfa47551a9b7',
     'beff4be2-9465-4007-9496-3990f0574b2c',
     'ae39e3e3-f771-42df-9c58-637589a31435',
     'bc48f024-2575-4d27-a539-7b2ff8ba5f3c',
     '4bd93dcf-bffb-4840-ba4c-72f24d00afe1',
     'cd8ca63e-9ea1-403a-b270-93bda97e65c4',
     '79641271-c823-4b59-a05f-912520e9dff2',
     'e6815810-38b3-4364-ae94-06386d89f761',
     '089ef137-67e0-491e-b52e-fe60aae93470',
     '5fe326c0-feb3-4f1c-9daf-22f7effc544b',
     '8dd9579c-6c7f-41c0-90b6-9ff88bf7d2b4',
     '85e024c0-4732-4407-9724-8472e6200e36',
     'c740b979-9c1c-4f21-86c7-08ec9ea775b0',
     '70340dca-cd35-4cd7-a243-7b487fe25507',
     '8d0af0b6-e798-4a81-bd97-b33049bacb73',
     '4f373008-0898-4283-be66-e9bf89e291c5',
     '7c7e05dd-5555-4838-97b0-47b2e87bcf3a',
     '395f6fe8-4338-4326-b170-523bd6022f07',
     '2aec5fb7-38ac-4b2b-a63c-b93ab698a299',
     '8aa3778d-f817-45f0-8a76-c611a9f763a3',
     '93023ff3-5edb-4105-b485-9001f3572169',
     '58f2e9ea-646b-474c-a41b-13ea684bb549',
     '4b88ca57-6b97-493a-8ecb-fc076e270e06',
     '7166c5e7-b7ba-45b1-9482-cc457e2b0f2d',
     '70e695a6-a6e6-4af3-a9b4-762a34e3cac1',
     '6e02fd40-6954-41ba-9f4a-1fb8d7de3499',
     '99fa3da1-4b61-4949-937a-0e7b5b454368',
     '4fc14e76-6734-49cf-a39e-22af44dbf821',
     'd1f06dc8-661f-4e3c-95b7-b35b94aa251f',
     '9d503a4f-132a-4ce9-bd6c-396ed842a02a',
     '016e2fcf-0cc2-42ad-844d-e1057b5081cc',
     '52377c84-18ba-4a8c-900f-39b8ebc6af96',
     '70187206-eac7-4629-89d4-8078b7b94f24',
     'c999ace6-0f87-4724-8e36-0b66aeab50e4',
     '499cef6e-d666-43f1-98c0-da1fb0192d3c',
     'a6276424-6991-4914-a8e3-f12778bf51ae',
     '611c21d6-ef3e-4b70-85ab-56cd98e4eec8',
     'e3cfa051-1108-4d43-b3b6-a254cd7a7c00',
     '53076350-30c8-4ffa-b1f8-57fd7e653ba6',
     'c84cb42e-85d6-407c-bbe0-6439056f6bb2',
     'c193ea64-02d1-42da-80dd-59e80c9d48c5',
     '4c7efa93-6edf-4ff0-a837-1b49e0371533',
     '8181343d-1439-431e-b509-a51048e05b37',
     'd1643d33-9332-4bbf-ad92-49e918b06f8c',
     'cc8a0fe0-e6a2-4b6b-b5bf-aaebe226b4e9',
     '55e7e473-472e-4a28-b58f-20b450b3fdcb',
     '998220c4-359b-4273-ae3b-e5e14c95ed6e',
     '883c0ab3-c04d-4670-9d43-252a525df513',
     '6225d3d1-bb4d-4831-824a-3a7fea6f43db',
     '8ac83900-28dd-4d28-a1e6-a7245d618949',
     '40f3353b-00c7-4667-a12d-7361194681a1',
     '18ae163d-cf8c-4d03-b467-aa477e8d0f87',
     '738c048a-0221-4bde-80f5-d8d6c4d2a906',
     '5dc810a4-e728-46ad-93ef-fc87806c3513',
     '493f9856-00af-47e4-9611-975be59d75ad',
     'fd089f86-db23-406d-a32b-c6b3e3e3079f',
     '75203808-7430-403d-93a3-5912fc216d92',
     '1ba5c6aa-490f-4953-a426-36b2c55fd548',
     '5faa33cc-2f12-44fe-8789-32b5065cf0ae',
     'ecaf8630-e511-44ec-ac2f-f55530c7e260',
     '28acb137-27a4-4d36-a3b5-103bed0ebb56',
     '9c80dc07-90ef-47b9-a7c3-ca8a2c88a1a7',
     'b1eb6772-de59-4d15-a2b6-520b1f010f82',
     '8c9e8443-2506-4f0c-b0bc-6186116771fb',
     '6c4e5ec0-db10-440c-952c-c1db8516a990',
     '1a74701d-a0cb-45ba-ac79-fb70eaba9676',
     'd0e4350a-d661-4bdb-9ca8-f6807733d3a1',
     '9d086003-07ae-4340-9d36-84074446fa30',
     '4c3821ab-b908-4ba3-8dd5-ab6b9994e5d3',
     '0b87cfa5-9854-4af5-a61f-e1aff5e06583',
     '78c022bf-f809-4b7a-9391-ddf2d198f197',
     '1176e5ac-7664-4cb9-b834-d383a12a3d46',
     '82913c10-a9cb-4c8b-bffe-ee35420da499',
     '4b8227ce-70b7-4eaa-a3aa-79156e39a455',
     '37b90d75-7b3d-41ac-8a7d-0cac72981a05',
     'cb38cb27-cb3b-4268-9e6e-8dc9860f0b91',
     'f554605b-ae4d-4f56-ac22-b6ababd9d497',
     'f9604081-e053-4bdd-ac60-a5003bf6db32',
     '699b5b72-6747-4e89-b107-048c4adcc827',
     'bd0e6ad3-bd8a-4228-93bb-6e2ebc04342b',
     '16ce31ff-ab83-4080-a785-67b2bf8ae86d',
     '4711531d-2970-48d6-aa0f-1a6d6d3912ce',
     'a0a2a14c-b346-4036-853e-a8fef3bd66c9',
     '8ea5ab02-e343-4fe4-89c4-90a3bcb87302',
     '40ecbe87-9bed-4bb4-b78d-705c6af273f3',
     '309b4f39-d1a7-4a12-97f8-2c9fa2ad55bd',
     'ef0b1887-6426-4b30-b14d-e49c2cf0e186',
     '04d9d1a9-bfa0-4ac5-a307-884e542bc58e',
     'ddcab3c2-ea9d-427a-aef6-11862e66a036',
     'd6c9d83e-806a-4c9f-85d0-4da715ed9745',
     '9d819b47-bf4d-4073-ad5d-222327baab70',
     '4de5a993-aa96-46ed-a02d-40aaaaa317aa',
     '96a1fe28-fcf6-470c-942c-695e9a9b7803',
     '0eca7c51-936f-4038-b369-2270b16d3d5c',
     '5d229575-0bcd-4ca5-abab-920574f1fd36',
     '6b57750d-5cae-4886-bf43-016344ae0d3f',
     '3252361c-e8bd-4f87-afda-37e7a18d8f26',
     '3c225844-c09a-4fa8-a5fe-f1dbe1a7784e',
     '437fd1ae-42ed-4038-9a2c-ab5a383fe362',
     '719e0dcb-7757-4480-af0e-d1f5f9e088a6',
     'e3b4a238-12ad-4519-a591-2b39cc8882a8',
     'de878405-e193-4c12-a837-6e114f96afa4',
     'b9745613-65cb-43f5-8473-b2ab09a5946b',
     '578421a3-f777-4418-910c-e0b8a2f77b59',
     'ba3742da-9d4c-42e9-a0d9-54657ab8c63c',
     '879cfe8a-9b6d-4d93-abf3-474de78e9bb0',
     'aff6956a-ae8c-460f-b2e5-e2ba175be6d0',
     '14326fe2-b717-4500-8192-d31f0ec449c5',
     '7d04669e-20a4-4b73-8b7b-fcf7d0a5f55d',
     '4dcb8d10-39f5-4ef7-af81-70fea2b1f4aa',
     '1f7e8296-55e8-47aa-b58f-09044a03de3a',
     '7cf3064a-2a0f-4589-a967-c33164124529',
     '81f6dd31-1427-4bf4-abaa-09f2ea49dd04',
     'fe54322d-c20c-4b78-85e6-d353b8420899',
     '37580ff0-3de1-4bdf-8af1-b9a9aa4ed8b2',
     'ceb5e408-d782-4b6d-bbd9-b04b88f5de67',
     '9db6897b-d8ed-490e-a59a-5f18879f5914',
     '7604dcb4-71c8-4f1e-bb72-bbe52f72b2b9',
     'e41950b0-6808-49bb-8f0c-70501cea596f',
     'e8b43407-6806-489f-aee8-76e567c7a647',
     '40f97fbd-9837-415d-a3ca-a7c407dfba89',
     '2937a575-659b-4ff4-a89f-2a5e0d08814a',
     'f891beef-098a-4c23-9814-e5b0f8192bb1',
     '197e66ec-cfb8-4470-bb5e-b846104d5ae2',
     '36feeecd-1fa5-4700-b8ac-7d3d1ba5752f',
     '6b3f1194-b0be-4e74-becd-163d204b0b83',
     '11d5698e-d021-414a-a7ea-375424906079',
     'a00a905e-90af-4f63-92b5-eefaf79efe4b',
     'c597e88c-6a9b-4bc3-912d-9d228105983f',
     '94e13546-8610-47c0-bfb6-8023866dc473',
     '9ec4564b-0c65-451f-b7e1-c1995510807a',
     '339ea252-7d31-4dc5-908b-c4a058ce393a',
     '86964a8a-2ecc-454d-861b-f2757bca705a',
     'f9fed3cc-936a-4286-b741-5d1992ddac18',
     '627d583f-e568-4fee-b5a8-f8024462f5ef',
     '19875fb8-2a05-4661-9ca0-8e3c321792a7',
     '6b7d3aa8-32b4-4fa5-b705-d080e31ca9ee',
     '073c1155-c6f1-451a-b22d-3508d041dce3',
     '60f51346-9058-4834-b375-c08f3dc4ba9c',
     '02d6a04a-63fb-4a50-8580-eef3c9a53ffe',
     'c40dea57-e2c2-4695-aa43-d8b94654f664',
     '0fbc486e-52c3-4528-a09e-c22ebaec255d',
     '38aeede6-ef0e-4f80-93a1-b994810177ba',
     'fef083e8-7167-4f42-82d5-d2e893aa408a',
     '131ced2b-5c23-4143-a11b-dea1a963ed23',
     'b27c4bb0-a7ba-4039-bdf3-915166f94b46',
     '4b6555a0-f283-4b68-b982-e105bbe77d7b',
     '2995b8ed-3884-4ccb-a57e-f1003f786842',
     'aacc0a3b-3d7e-40ae-aefe-2591222b075f',
     '9284a0e5-3769-405c-88fe-9c30fb39c49e',
     '311b0c23-af82-48ac-8ed9-607b9ca34ccf',
     '07f3861b-76b1-4e68-9fd0-ffd825ffcf33',
     'd47e5694-3916-4d4e-a74b-c2608bb29b48',
     'bb3d0834-83be-4158-8d8e-6230178074a8',
     '1df24e31-3065-4ec5-bdca-3158222d194b',
     '84e25730-d779-4f3f-9cc4-1e0a57380c58',
     '2ec781e1-67c7-4f28-bc26-1aa46bd20f08',
     '4777e4d2-ddc6-43c6-9918-d3e1412632fa',
     '5148a28c-185e-420e-af91-2b340d5c56c1',
     '23c91fe7-bcaa-4335-9c33-bebf74cfb3de',
     '470d4fc0-f19a-47b0-90d9-d51c9db1a2fb',
     '930a4ad2-04b3-45c1-973d-8e09c54a6f6d',
     '65ba7fff-adf9-4e17-afee-b08181bf5b35',
     '4da98a11-1c77-4e42-a5e7-9c731cef3e0e',
     '14cb29d1-1ddf-4bfa-a092-0f196af250de',
     'dacad6bf-c434-41c3-a6dc-6e714e8135e8',
     '497e7247-caf0-408f-8325-5c20e532dac3',
     '065cdb33-9229-4886-a547-0f2927f92783',
     'd9672c2c-ada0-460d-8fe6-67e07e948607',
     '1f711810-5365-4ff2-9210-5a36d2d8f19d',
     'fafaf1de-f198-4251-beb3-1564ca481dfd',
     '5f1759e8-6625-4fc5-ac87-1889b4d86876',
     '165bb81f-8eb2-4606-a632-f6d201837a0e',
     'ab6f1774-ed3b-4952-b23a-1659176fec5c',
     'a62851d0-7cfd-4fcd-8353-5f2931be7ddc',
     'fe5182ec-6732-402e-91f6-a5ddec5107ba',
     '5608e2f7-d0c9-4383-a4a7-35415a22aac7',
     '17c657d1-b46f-4210-8bce-0dddc75aa486',
     '273fd020-5a5e-4ebd-954b-62fb11a91fb5',
     '2612708d-6d84-41f9-8047-ff233febee1d',
     '5daee5ac-e46e-433d-92cd-aba5a646548b',
     '3a38e46f-4ade-45cd-883f-e84617534371',
     '4eb7690e-2297-41b3-a95c-cdd157954642',
     '040852c0-3b9c-4fc6-8229-d7ba8264e9ce',
     '1974e2b6-9ea4-4716-9cd9-3f6a5649db46',
     'ec552d88-c61f-408c-af69-65dde2844512',
     '7474aa62-b9e3-43f3-9832-372bb318cdd1',
     'c5ec3e1f-861a-46fb-b22c-e7d7de2b1922',
     '654be316-c8e8-4bff-b084-5dae1f991920',
     '5248266b-f72c-4a62-b311-3e47e54cb883',
     '44c4b514-7b57-42eb-a4f7-defa1fb249aa',
     '297411a4-d038-4fbd-acea-5929c8f9e921',
     '5b045e3a-51ac-461d-9a20-a6c391e8170a',
     'c38bdcec-1224-43a3-9567-a0539b515537',
     'a723f190-7f19-4d03-9309-8d864c26f2e9',
     '288cb6e5-e750-4164-946d-f72bdd63ae74',
     'e4fc6b82-b583-4ca4-a4b9-48f239c1ea9e',
     '4fa8ee78-56c7-47a8-b29f-0d3884db2549',
     '513294bb-8a03-478e-955c-23f6091e4cec',
     'b0fbd413-db3c-4e57-8d27-4dc0a5725661',
     '20b07e14-d5b9-4dbb-99db-e0b784b045f6',
     'a7f5d963-4717-4a28-aa09-6feaebf84bf1',
     '4f10509e-4e9d-4cf8-abe8-c21b2658f0aa',
     '93cc5682-c2bd-4ee1-b0dd-fc4e62ad4ae0',
     '3d50c5df-bfa5-47bf-9664-c09c8a3a34e9',
     '041a771b-41cc-4b36-928d-ec321f918db5',
     '162b7c19-331e-4281-a521-5e6c5f8baae5',
     '94b84ed4-61ad-4364-b98a-926383362fb2',
     'b18fc259-43f9-451f-bf0f-5dd5fd77c535',
     '0d64540e-c1e8-4eb4-abc7-e623f68a97ca',
     'a39aa1c5-ea8a-42ba-9299-9e76cebe1313',
     'c2e82cf7-a9b0-4817-b77a-a225561b24a9',
     'faecb7cb-6b15-42fc-b0fb-ef2b512c751f',
     'e168a7ca-ba60-475e-a879-0ac979014fd1',
     'eb6ce13c-8779-4281-ba39-deed0b889ce1',
     '780b28fb-47cd-444b-a320-6375cf9329c4',
     '92ea331a-ce01-4d9c-ab8b-6ca9f089b6ac',
     '8c488e68-7efe-4d3b-b11c-b84b7c3b1735',
     'ccfb8067-9cfa-4180-be80-bbf96106ad96',
     '41298b0f-7dd7-454f-b70d-0158babac7c1',
     '2e7968fe-6970-4a4e-b800-0d38c88ae8e3',
     '8883ba4a-adbc-463d-8df9-cde5aa36a0db',
     'bd2e778f-8a57-4247-bbfd-76f2231fee5c',
     '954a8fb7-4c39-4fca-b790-b246b3c1aa5b',
     '5f93a82f-e821-479d-9005-c0d2c9226fe4',
     'd950144a-4a43-490a-8323-e0564698208b',
     'eecac9d7-322d-4b77-8531-5aa573b8ed18',
     '99ca47e4-9d93-4280-a2d2-85268bad236c',
     'f8d3fa19-61ea-485d-8220-5c55487c2c31',
     'ee613842-fa02-42dd-800e-c93a4cb48d46',
     '57d6907f-63d8-4900-a571-bc90456cd1d8',
     '6d364953-7933-44e9-aafd-f51195a44150',
     'dec93d17-0d97-4384-9ad1-f462a8b12ee1',
     '197d9f4e-b8ad-4b5c-8024-34e1f54a94ac',
     '52748e18-912e-4333-a972-7df6729bfeeb',
     '40402be9-df87-4cd6-8928-15f50805bc0b',
     'c1fe5f36-acaa-4f8b-aa70-747355dec3e9',
     '02f0e278-71bf-4b08-aa39-9998e981cd20',
     'ee6ba316-e02e-458b-b0b8-a6935bce3b02',
     '81cc6263-800d-47a5-af70-f652dc11af18',
     '3a8726c9-1529-4abe-9f79-ba32fefe9240',
     'bced2780-079e-47a0-ae0e-d461e4bc694e',
     '514fadb4-f5d9-42c6-9e8a-62b366528365',
     '6a67118c-763b-4d96-bc16-6bd2d9e4a181',
     'd2d72e43-fd61-420c-946d-17a38f7c6b01',
     '9acaa4ec-4370-4137-a67e-ff2cb7aa145b',
     '5225b5ca-7265-43b3-9a73-da4839aad343',
     '3f575ab1-70b5-4f84-8f16-6dbdacaa21d4',
     '0a6a20c0-bdb1-4033-a073-2bbab0c3a3ed',
     '01805fac-7187-40e3-8a7e-1dda3c6c87f2',
     'c9e40dba-a483-464e-97ba-33dae03150fa',
     '00a53a52-aa0e-4ddc-afa8-2c50c21dfd03',
     'd43cb265-539d-4815-8d2e-add7b728727c',
     '6b86866b-affa-42c6-852d-6e9c86b59e41',
     '68a895bf-7713-49d9-9766-8873435fdd46',
     '5861f6f6-ecbd-4476-aeb3-2429fc75b206',
     'b771077c-cbc5-4fd0-9d2d-5dfee37ecd22',
     '94a09f74-2d8c-4379-92ad-7fe5b02ec188',
     '564809f9-883b-4dfc-a9f9-2db11f35e903',
     '988aaef5-55d8-4efd-a2aa-8a17e687965e',
     '594e9f8c-fcf7-4487-a945-bb7dbf3e4918',
     '4b52e603-0b9d-47b9-b773-6c72a124d3ed',
     '10a5fdf1-cb63-426b-ad41-6cfdf66b1c52',
     '17d4cde2-f6e3-4d37-9658-ca054cc406ec',
     '2dfc80bd-d515-4d75-9316-f3ae6c49ef40',
     '6dd0a001-2384-485e-aef8-5c8b613a8c70',
     '304d0cc2-0173-4bb0-b7eb-a77aeba4246f',
     'd00fedfb-fa0b-4005-a5ad-5dd487a11919',
     '5b1ac26a-c8c7-4a5b-a3f1-97dd14f920d3',
     '2f1ff843-1c12-4c6b-8ddb-389e95f5c7e0',
     'a182ea02-9438-40c2-a5cb-ef66c9e0d0a2',
     'd7669c6e-b08a-4c01-af74-7a87e1642378',
     '71f93c3f-8d70-4264-ac60-ae72b8e42d39',
     'b9b1abc2-8934-4468-af82-68fc14534784',
     '4232158c-63bd-453b-bd16-3a635e3a1be0',
     '5fedc5a8-7617-4529-900c-178cf3969d52',
     '2a6670bf-3edb-4850-b36c-3d93041c6ce5',
     '8d01149c-a443-4812-941c-4e56f36625d7',
     'e0664c10-3aa0-4351-8d2d-a0c8dae56dd5',
     '53af77c6-6730-4c74-beb0-f717439e6e5a',
     'b7e4a296-bd3e-44fb-b0dd-0c008ed0e2eb',
     'e801eb1c-bc55-4150-849f-faabf51dc54c',
     '9b4b72d3-4b08-48fc-8502-53e08bf718b0',
     'c456eb6e-775e-446c-bbef-d0c49c32a4a7',
     '67688b1e-869b-4e14-9a9d-7babeb798478',
     '2fc4946a-75d8-4493-bea0-f40afde6e54f',
     '175dc771-9442-4269-9507-9066fa4650cc',
     '5adc2624-c2f8-4560-b23c-cca9f257c35a',
     'a3588e56-cfb2-45e4-af3a-17137700ecb4',
     '5e00e3dc-270b-454d-9f8d-339d8fbb69dc',
     '75700c66-c480-4352-a418-bef060a7347f',
     '134b87d9-40b3-4a87-a966-86c514bd5602',
     '2614cf92-1523-4e4c-88fd-58687208f371',
     '5e557078-7767-449d-8a32-d96e63b86de0',
     '2eaa4c0a-5d55-473f-866d-78b9bf3eb5f2',
     'e02dd936-91d5-4cd2-8bcc-d15a56cf7400',
     'ca79f9af-9367-48fc-95b9-53175ba79c37',
     '6bc56d36-0fa7-4910-bc2b-52987301018b',
     '2d7de8ce-4b62-40d0-b2c8-6cf39d8c06dd',
     '308573f4-228a-4302-afdc-508b68bc970a',
     '1936cda8-c56d-4437-953a-df6718e0f4d1',
     'cb17027b-c85f-42b9-8afd-855f5e96663c',
     'af48541e-53ee-4c3b-a73b-ffb93dc881c5',
     '852defce-8ff0-4ef9-9068-9b4138f2d94c',
     'cea540a5-cde8-4749-956a-888e4335efba',
     '344c16ed-4315-47ce-a68a-fbbf58507a6f',
     '479eb527-639b-45b6-8ec8-504a56d7d51a',
     '33d21fcb-bc6e-4146-ac50-17004c44ff5f',
     '0f6a6ea8-cef4-4201-a8cb-c7b084f3485e',
     '91d84b86-36b4-4bba-8cb2-b70e2c39aa6a',
     '8ec8fa7b-7048-42ac-a9dd-15f91a07b554',
     '59e8ae59-a35e-4acc-a2ee-30d0655d63e8',
     '7df32baf-8d38-4223-9390-49aab5973db2',
     '93b0f6db-34e2-4a05-9f9c-98128cfbabf7',
     'cce1092c-18e7-4698-8494-7e7a6ae5c695',
     '61bcc823-0b95-46b2-85b4-6e688cd0a5eb',
     '0a8f8e66-5a3b-45c6-941b-ea77d27fd63f',
     'b2131618-3e6e-465a-8ebc-588dbde4e8a2',
     'a25014c3-2a12-45ea-965e-4f52fb568c49',
     '4ef100e7-a7e2-49a7-9f52-0d7faaf9863e',
     'e348792e-50fe-459e-90c1-95b003641978',
     'dec5f39b-417c-425e-a9a4-b884d2ecf9df',
     'b238eed2-f485-4e57-a2d6-18ac993952ba',
     '600f8379-2f63-4f44-8eb6-1762583d34ad',
     '185b46af-a097-407b-a6ca-41111b5dd8c5',
     'f8e13c27-7474-4855-86e1-73a93cee925e',
     'ee835340-8bb4-4079-a56a-a60daceb092b',
     '7d79e165-d339-4d8a-a9a7-93b525594926',
     'b4929f9d-cc2e-401a-9af5-23fa9441a009',
     '920aca12-470e-4297-8bb5-98fce927f0e2',
     '74b7daa3-3971-4d6b-bf30-88b178e81163',
     '70afecbb-1632-4ad5-9a57-df7338c10fa1',
     '4c7226aa-f420-4d3a-8d71-5beb719e4cb4',
     '26bd4867-1e47-41b9-a445-8dc78a9d3e89',
     '13c49279-9c80-41e2-b6a2-1513aab66300',
     'bc098b76-55ef-4f31-9300-00e6099a8e9f',
     '8435c505-a8b7-46ec-a379-6107b98975d6',
     '996f6feb-a744-4ea8-a779-3b9828051d21',
     '3709ffef-b87e-40e3-b193-a9a3c713d9cd',
     '4ffecbfe-c00e-45c1-94d0-a14dd52d26b8',
     'b32a9481-8894-4d00-aa91-b8d1f67e2248',
     'e07dea39-fd85-4ee8-951f-8d0e81a3debd',
     'bc3b2cb8-d7f5-4167-9783-c5e8f75a0f82',
     '3f8bf65d-e966-43c9-9990-40b00f0e7f88',
     '5113ed43-1e87-4dbb-a5c5-79107a25282f',
     '9981ed63-9aec-4ffe-aa48-f2c79009a2ef',
     '1eb7d18d-33bd-4291-9f22-a5531c868044',
     'eaabf304-6cd7-458d-8285-bdf06cb8ebd5',
     '6fe13614-68c3-47eb-9251-3d45afe8ebe6',
     '98fd92d9-60eb-413f-935c-18b21fadf96b',
     '9dda11c0-1985-4750-bbfb-8e9c14efdd91',
     '5431687b-78af-4a33-82d2-177263d0812b',
     'cdff841b-378d-40e7-a2da-c64c3e967d5e',
     '2f529bfc-4eb9-4290-8e7c-6aa737e79a02',
     '87a2846c-5ad1-4ac1-ad40-46cf881fb13e',
     'c6c99eaa-6274-4b1a-8e0e-47b11af78c18',
     '8b513e44-950c-4946-96f9-fd64cf412efd',
     '85c88699-88c8-4566-af46-fc92d9fc03e6',
     '920a0c75-6a6c-4d60-8706-3f2bff322800',
     '90152eff-7f00-4ea7-9ff8-ecfa3ca5b331',
     '4c6af170-e6b4-4791-821c-9f01efc4bc11',
     'f360e56c-44d2-4f4e-98e0-4e425d77d507',
     'ae0ff1fd-4958-4cef-b57b-9d7bb5fb0998',
     '6bf771cc-98d9-4d66-9e95-3d5d794088e9',
     '14831cc3-24df-4021-ab13-6962fdb646e7',
     '5b8ceb46-121c-43a0-ab0c-35c5a2206228',
     'ad3c8d11-b356-4b49-b6d1-1f9329b82a4f',
     'a0bddad5-f30a-4669-8252-81a198fb5f01',
     '1a9df18a-84a5-4fc6-b3c7-eef5fd3fc623',
     'a34d83d8-f5e8-41e3-a0d8-9b1f81b6b263',
     '953768c5-32fa-4190-8fee-1da82d8710bc',
     '1df001b8-a5c9-4ab8-b6d2-173819046515',
     '54c037ee-abce-45bc-9647-992dd322e294',
     '1a9153c5-3b1e-48fb-8db2-41a2de91d3d2',
     'eed4f6da-086b-4be5-ab62-9536866d86b6',
     '89f364a8-6a52-449b-9703-8691b79a6f03',
     'f78feea8-d6a2-4e14-8921-46318893a1ab',
     '3f01aaeb-e4e7-4526-87a7-b1d656514f84',
     'f4f342ae-9eba-46dd-b6e8-909f27a26c47',
     '9a73d013-b586-40d1-b8e0-57312b13f598',
     'd6d5a2c4-a2ab-46dd-9f6c-eb7a0e58ecc9',
     '86aec3d8-bdf5-4b94-89bf-75027a626e22',
     '65f9c9e4-9fb7-41b8-a914-e9df4cad3556',
     '5c7d777b-45a7-4236-89fe-2f9861f2c0bf',
     '9fda2f36-84dc-455f-ad10-93b2c9720a57',
     '4c647bf3-034a-4392-bd99-ce89315a4ec5',
     '5679343a-0074-485e-b581-50c686e07df7',
     'c72c5a04-a6d8-45f3-9343-8f5cbdd2ba55',
     'd559248b-08a4-460a-8510-3f516deba921',
     '814d4ae6-c978-412e-9328-6db228688bfd',
     '7500f36f-bd3f-4c70-bffb-6a961483bff3',
     '123dedc6-dd89-4e59-bccd-159d458aef0f',
     '38809184-db84-4042-8fe7-8840d6ffd1c6',
     'f41534ae-0785-4cb6-92a3-1d457208ff28',
     '34f9de03-1e4f-4703-b619-15c41718a7ad',
     'd41d31f4-5661-4c1a-a1e0-231ff9bae44a',
     'ec963de3-4d24-4cd7-be14-f8751dde31b5',
     'c985759c-1ed2-4964-8c29-e17a0844e44a',
     '32a0c3c0-41bf-4fcd-8668-3071bd775366',
     '0aee03f6-85a4-4f21-a80c-01d4ee3fac14',
     'db24d53d-40a6-40e2-bebf-cc72604cc382',
     '34383adc-0a50-4373-a0fa-8854b479ae48',
     '910ff7c2-3ff7-4ebe-b912-cd04f555ae41',
     '5e037301-b06b-4b91-896b-fa411d055f7e',
     '5b4a3e41-75c8-48e3-be17-050d8cae6eb7',
     'bf24856e-b682-49e2-840a-e2f047d6e163',
     '0cfdf707-e4a4-44ea-953b-450a6a53a55b',
     '46b4a4d0-5c47-4da7-a002-e7767af0455f',
     '1e9e611e-fd5b-4d40-b1c0-2a9a00583e89',
     'f4a10383-174f-4ca6-b7a0-1150d9e591e3',
     'bef78922-5f35-4b0b-b66f-867376c44d6f',
     '91f060a9-d6cc-48e2-8ad9-362ab1b25fe2',
     '8469aae4-4daa-45ae-b704-979820e9fd7b',
     '24e2c58f-6c2b-4586-b112-3d047d1182b4',
     '72d650dc-dbb1-4cb6-ba9f-7387ac0070f8',
     '4d04567b-0a60-4707-aff8-65bd76932896',
     '0790387e-e285-442d-b660-a004d0a5a16f',
     '752072ec-1a38-4a78-9701-f63f3c8cb3a2',
     'c35e79d7-7132-4537-90ea-360e6ff9504f',
     '6c66b350-ed7e-4308-a217-4d3d4b663a12',
     '532c9ef0-9898-4e52-a46e-b4ead7493fb1',
     '86344652-e67c-4176-a415-0839cc2e434d',
     '9a728695-796c-4e6f-bdc2-96c5bd64fbea',
     '78ccab12-8733-4772-a2e0-17c7e68a69d6',
     'fad1ad72-fcd9-489a-b07e-1951bc19e1a0',
     '06da057a-0980-4de8-81e9-897af6212678',
     '5098bdf5-4c55-4c17-be39-21bd88ad1a2b',
     '9cbe5aae-cd72-4315-8ef2-7c6dfbb3b9b3',
     '827aa325-a644-4598-aa69-4319841875bd',
     '4614c267-d2ba-4373-968b-49e0ba8cecda',
     'b356e548-388e-4062-a3e7-9397703f2183',
     '4f025cde-00fb-4c70-907f-5106446d9b5b',
     '99dc401b-87b4-4bca-a87d-9f9049672d0e',
     '835ff6fc-5a7f-4ec4-9726-1b8da712566e',
     'cbb9f8fc-d9bb-423b-8050-41022fe1efb7',
     'f06e96cb-cd66-4571-a12a-72e26c23d871',
     '5f7aee38-fac6-4ddb-9c0f-38a5e2706d66',
     '337bfb43-98da-411e-8b9c-ebafd73ddd31',
     '42ead4d4-d195-458f-b3d4-60756e523653',
     '198a38f3-1cf8-4d24-af3d-30417f8adad6',
     'b8d212bf-8c9d-4368-b5d1-87c859ed8766',
     '3f29634b-ad6c-494a-8418-163116d6e327',
     '2fbe606b-83d5-4b7d-aa0b-1eed516b4f41',
     '358dde6e-b5ae-4fa4-898e-05cfff311129',
     'e7dbb0ad-caf6-4440-bb48-d646aad3cb8d',
     '7244914f-524d-48c5-a08b-f2be96408fab',
     '1bf8f1f2-fbdd-4e08-9f12-daa182d154c3',
     '0efc83f1-e1eb-4080-a7b0-9ce092dfc211',
     'e1bec8e4-fecc-4d71-ac91-2b88f4ab5f40',
     'ee2278e6-263e-438d-89fc-00b8f9eff85f',
     '8bc5435e-0acc-4480-93ab-1b525a75c79f',
     '3ba3aaa3-202a-4ee9-8698-1d831c748364',
     'd5f9205f-d28d-474a-bb45-f6723b33866b',
     '3a38e166-0ef2-4df3-831a-28c0135244e2',
     '9a8ab2df-ad4b-4779-88d1-afabba18800a',
     '82002781-ff80-43ce-82a1-84f135f8387c',
     'a3ac1e2d-7dc6-4ff1-a8df-bb6ff0f52e67',
     'fab767f3-06b1-4694-8148-eb3b58b05099',
     'cf9a4439-3abc-4289-b2b3-e59f825773bc',
     'd9abc8e4-707d-47f1-9ff7-62d5b7d0d34a',
     '137a4084-3a85-4f71-bb4d-48285f770e51',
     'def723da-d2b0-4907-9575-7c2e2640da7f',
     '53c08c3d-b68d-45a0-805e-3b0db297ad5e',
     '721e4663-3512-4a43-92de-2b542d3f56f2',
     '7a6bb46d-853d-4a49-ad6d-be434d9e8214',
     '9c4008ad-9e93-4205-b748-5655f3ba1e82',
     'ccbdcef6-89cf-42c0-b521-26b93e6b6bfa',
     'd9bca1fc-1403-469f-b043-9d944500a6e7',
     'd3f53b19-5b2f-4d6a-ae63-1e3cafef5991',
     '6ca12c7c-cd3b-433b-92c2-02ccd38a675f',
     'faa24a33-0b54-4cc4-b8a1-03fc925b8616',
     '5f386d96-6fd8-4eb3-8acd-8fba08721cdf',
     'd2421267-c56f-4494-bf03-5833a0109ae0',
     '8e7c512e-63d5-496f-a29c-716bff78134d',
     '9d575ae7-83ad-4087-b0fc-76997eebd13f',
     '11e431e6-8a15-491b-9185-134929ce310a',
     'e8795ae1-7969-4640-96e3-049212087e52',
     '85d5058c-53e9-49b1-8475-d3df2336b0ff',
     '5f6e4983-9173-4b7b-a93d-f713963d99d1',
     'ca4b51ba-988d-4e5f-9522-ad45940774a0',
     'a9aa34a9-026d-40de-be42-1d8064072a48',
     'dee74669-b1ed-41df-bf06-177c5766071d',
     '4cb55c03-c2cb-4d22-a19f-cea62756b102',
     'fd2a1c41-0bc1-47c6-8ab3-d238a9a648ef',
     '991f1415-21d7-45f2-8352-4c24463c3464',
     'fa80fb1d-b065-4849-9aac-1daf8030bec5',
     '5b4ab2da-021d-41fa-9e41-26c9fe619a91',
     'c7bb202e-6a69-4a84-9814-e55e6bd6d54b',
     '740bfd32-e162-480f-b824-e4877ef945c9',
     'b9e12a33-e1e1-4561-bc71-d94c1716c5ce',
     'f551c8be-5eba-4562-af64-69a6adf52705',
     '8b0fa1ae-3abb-4131-84f8-a7013ba75e34',
     '29d93eff-7448-4107-a203-08cd3471f2fe',
     '8aee549c-5e83-4290-8ad6-2b963120494f',
     'ebd864a2-0af6-43a4-a47a-e021c70dc264',
     'f5b84411-fc45-422d-a9d4-44fa4279b6d0',
     'd07d077c-0837-4ab0-9585-780309a223eb',
     'b9f50cbb-336f-4dd4-9c63-435d336398d0',
     'fd0de4d8-f424-4391-a7b7-6b1b827c1257',
     '0a60d788-270d-4ded-9c9b-3bb0da080d04',
     'a008d6ca-8b0f-4e9e-8bf6-42ed8aa7b4c4',
     'eb40e0e5-0227-4e52-9574-c4a79f59a2eb',
     '7b650edc-b51f-4c0a-8737-943774e4a710',
     '7535ba88-d131-4eee-b392-36f5e11c85ce',
     'cb0295e9-fbe7-4fd5-ada4-eeec6c754043',
     '0e0a4b45-657e-467f-a409-ca92dbdb94f9',
     'f31a5812-06e9-4a12-abf6-9b464ef5ff19',
     'cca42ae7-fbb8-4463-a3ac-b231514f5cfe',
     '7e7016d4-1006-4fe0-b193-8fbe37e27f1c',
     '60aab6bd-97db-4d4d-8f7a-a4cc89d2cd1d',
     'a51af53e-6e4c-4651-832d-e7349e808a6d',
     'd1f09e66-6592-4fb2-9589-b8ae4edf93c0',
     '5e4b7326-0a56-4ffe-bd53-2db13625527b',
     'b14d84c0-b2e2-493d-83f9-7827d11fb214',
     '3eb4f7c3-fab5-4e13-a1e8-e072d55efdc3',
     '968ac323-b290-4105-b0da-e32be8d20cdf',
     'b7f09196-9615-48fa-88ce-939b56d9d74a',
     '3a408851-f3f3-4d3c-b465-15253d1c4add',
     '69824590-c804-4192-9fcf-9351613a158b',
     'e57537b6-e1e6-4881-a9f8-9c660d4808e0',
     '44061a72-233f-4cdf-b4de-6bae342b0117',
     '76af68d5-e40e-46a3-89ef-3a958fc4fbd4',
     'a7c8635d-ceb3-4dc2-a1f8-938fdeeac2ff',
     '3ed7d780-c857-486b-ae3f-38ae3a36aab3',
     '0f748037-90a7-4a1c-9564-7744004198d1',
     'af8dd73a-716a-4902-bc01-88e8e00df672',
     'b05608c8-c017-4866-a214-5ee36a83fcb9',
     '9ff6bab6-169a-4efb-b6f5-55bb4e6e9b05',
     '8b17f185-bc09-456d-9043-c653681ee583',
     'cac17be2-db9c-426c-8a33-e97eee8bb836',
     'fdc4706b-6963-41e4-9a96-ce7b44b2dc46',
     '8c5e85b5-3645-4181-84e7-6904cd6ebfbf',
     '357c9029-1763-4206-a005-12ec72f49273',
     '39d37139-5684-4927-bda1-173200f2df45',
     '7b001cae-b5d1-4d8e-82e9-c5accb7afddb',
     '97a00d67-74fc-43ed-a057-c045d2901959',
     '8f3faf8d-8c56-4587-81c4-d6eb2cc5a4c7',
     '5809f204-48e7-4dff-bc92-d22986d01459',
     '615748d3-3c61-4109-bcb7-912115d97be6',
     'fbc018fd-291a-4a71-9c94-fe71ce823ca0',
     '78a6df60-886f-4139-ba28-c0ad7dcd8af4',
     '7e939a6c-0a02-4799-9b3f-1bfbf0a76adc',
     '52e06937-6b52-45f4-8ad9-974a9efc0df0',
     '2fac0ab0-cfcb-43c0-8e38-d37ae54b3113',
     'e6cbdadb-c6f7-4b7f-b03b-9ddfb5aedf1e',
     'e11a5b5c-08d3-423b-b209-fcb1cc7e08e6',
     'd9afcfa5-9199-471d-bfd6-ebcc2486aa6e',
     '42b9ea17-f4b7-4170-a6b8-61cbcab6effd',
     '33106cf9-22dc-45d0-847d-8465c6303a57',
     'fb6fce0b-1ff0-41f7-b784-6c03bb2b17e6',
     'c7d75d22-9225-4bed-abc4-b9bac29de4bb',
     '485190a8-8d92-4f26-9bd6-9ce41d9991ef',
     '6d4d8b9c-7d84-43ae-ab2b-e5adfb986a1a',
     '0e9e7143-75fc-47c4-8f64-00dd5ffd0146',
     '23c0797d-36ef-4dba-bcc1-5f9f8f38763b',
     '0f164232-2949-48d1-8558-91dcf9023ce9',
     '49b169fc-a560-4aa3-94a5-209bbe40800d',
     'd9dfd7fa-cdfd-4f5f-a887-f938ebdb18d4',
     '275b4edb-e632-48f1-b476-1e98833aa148',
     '3f695b94-3d36-4c1c-befc-43cd25ef41ec',
     'e1997e6c-7338-423f-a4fc-c25b6bd4d337',
     'cf8a89c5-e36d-4390-959d-8e4d7b5e9e8c',
     'df3d5806-6ccb-4711-96e0-eb91f30b6c2a',
     'e6af885d-1482-40b4-91bd-67048726f729',
     '663b6644-62a1-4c7f-92b3-ea328ae7a30c',
     '42b05cf7-78a6-4c0d-82f6-17f4f6548e83',
     '00fe5bb2-6b64-4a36-a706-eac4dc124ea7',
     '6283ab65-ed00-401a-b069-4fc094184c8f',
     'cf3a79bb-1f09-4d45-a9ae-ac6f2da6c16c',
     '9a897fa6-c51e-472e-9f14-172e0cb339df',
     '5febea77-bb32-4dce-aecb-dac8b050b5e0',
     '02b7468a-2c49-474d-8c6b-3188904fd85e',
     'edf4bd38-1baa-43b5-a04a-074bf48313b9',
     '7b7a965e-de91-4b5c-b202-13708736c164',
     '7a3ce6b2-e90c-4b86-8b3c-5f3a3d321d27',
     'cbe8fdf5-ba7d-4fcd-b4ab-3cd7a9e03f43',
     'f03a01db-3399-4a1d-a4fa-2e2dd8ea612f',
     '20e49198-23eb-4330-8a5a-622754c7d65f',
     '86de5b46-bf9e-4599-b49f-5703a555444f',
     'dc7b7beb-d4c1-4916-99e4-344709121030',
     '889c4a38-2b90-4a65-8399-68268cb28928',
     '774ccb7d-0c00-40dc-a16a-aa5bfb1f962f',
     '04e5447c-0f3f-48c0-9ba8-5e7412d7c098',
     '0fef08ab-4bdc-418a-ae22-f992ec4f7702',
     '34716706-ef50-4c62-bb24-d8e2f167a4a5',
     'ce4be29a-7c0e-4d75-93a6-4452c19003c7',
     '1aea0d20-fd7c-433e-a1dc-048c68ff6178',
     '20cd601c-dfbf-41a2-b631-dd15f8a42641',
     '3f70af22-08be-420c-b920-51d0ec164305',
     '1dbf5dbd-90b2-4162-985a-cd71d26a3dac',
     '64cba20c-e41b-4ae5-9650-150e8002a21f',
     '197847ed-5207-41c3-9426-cabbeb54add0',
     'c916624b-9480-4916-b7e9-9330d75a07d7',
     'c8075223-472e-438e-b1a1-ce4c1b67f431',
     'a93ed767-e7d6-4d29-b296-94519f37ae54',
     '36c01da4-ac69-490f-98b8-85c8d7848183',
     '80d892f2-a5b2-40f7-bc99-7fce2927c050',
     'cf0a2657-677f-4929-a2c8-a23f5880259b',
     'ba4982b8-97a1-48ac-b6b3-84eeb1ff5e77',
     'ba593149-8b3c-4e1e-8df9-2b470c49a4f4',
     '5ebbeac6-de5e-40ca-a390-29004251c918',
     '0a1c6dbe-74fc-4984-9721-f301535753a4',
     '47399c23-4d05-45e7-8aec-a6ae136689e1',
     '1acec420-e195-4556-b6c8-a424596b23e2',
     '14f4f107-ea42-41e5-815a-a7ef8db92cd3',
     'd3da6d82-4012-4c2c-a7eb-152f0ec23947',
     '308a0e52-a9e6-4075-a807-23efcc16337f',
     'bc74ae3f-e221-4abb-b068-3a5b03618aac',
     '54d593c4-42f4-4476-aa32-92093979d1fa',
     '767e793e-dca8-4a49-92c8-2cdfa6a13125',
     'e71400ab-ca29-4b25-87d2-bddb4f1208ec',
     '6758bfb2-b34b-4a04-93bc-6570934677e3',
     '6356640f-aa58-4e2c-90c4-0c4c0210befe',
     'eefe5e34-9386-4dd4-b052-f518344fb927',
     '5e86a4eb-d253-4f2b-9d36-811a38818609',
     '987de286-9e47-48fa-9fb6-c085f6e228a5',
     'ada9b3bd-a10e-44c4-a6c5-8eebae2720ae',
     'daf9989c-d0dc-42ba-9a14-2eaaf1c7d7f9',
     'c6f67354-5c46-4bbe-a514-48cf0b1f8208',
     'a21ac54a-e190-486c-b805-c83b89589f2a',
     '82a3e3eb-3f80-48b6-a55c-986cc1d0a709',
     'f3a0ddf8-a396-4a4f-9d01-347097f0c187',
     '6d9a2266-139e-4802-bbf2-e19c109cc10f',
     '9cba7a76-428f-45f6-aa2f-2c2e6fd5d342',
     '045d0e87-585c-4126-a2c0-c744effb11b2',
     'b406a439-bac0-4fed-9733-94c430be44d8',
     '77745a3b-3ac6-4a33-a111-3c657d267e4a',
     'd92a6267-7fbb-4b28-846c-8f32228c5e0d',
     '39d72bff-e5f8-4fea-a00d-e07a2cbc46fa',
     'a5056981-e12e-403c-9bc8-222fc7efc81a',
     '2b52c7c7-844f-4e84-9007-203a1b72e96b',
     '08b8a222-046a-44d7-a609-9e2ba5dbdec5',
     '7586f20a-8b0e-4467-8280-f68cbd64d1b0',
     '872878c3-0678-43e8-87c4-2bdd6dabd616',
     'af94a4c0-39b3-45ab-89cf-c7f246155b3a',
     '7f093283-b382-4fec-ab47-98d83736a490',
     '11eabd81-c391-4201-9c16-166286778096',
     '0c3da7ec-5600-4e50-95b1-1b0c26ca64d6',
     '46d0d860-4427-40b0-bbac-e29317d99754',
     'd0202263-80ce-47d3-b257-177d354a3bc2',
     'b69fe0a5-6ce3-4680-b2ad-b1212bbb2861',
     '43f61bdc-b6ee-4a7d-abf8-bb61062dfe62',
     '11313a84-d5b6-4553-8386-5e307edc4081',
     '59b07328-38b7-4b5e-93f2-66e5a0e82fe9',
     'f5176721-ed07-40b5-bd2c-42f6ea7af975',
     'e09abaf9-c6a9-47ce-8b3e-a6fc1ae60a62',
     'c6084d9a-5010-40c9-b04e-69a9bf626af1',
     '632f90dc-80f8-4f1d-92af-ab39fe662d7a',
     '709ced54-a3af-4a6c-b28d-977589b00585',
     'bc1e9292-5a5e-4fc4-bf8e-5ed0e7b1152d',
     '6893aae9-5c6f-4813-b228-2770aeb4195e',
     '10dceb32-9d5f-453f-b17f-71a41057809b',
     '4680cff9-c0b0-48c4-bf99-03ff8c189145',
     'ade53c84-f641-46a7-8789-8ffaf230bd25',
     'f2eb2e96-b02d-4eb3-9404-343e3f4d9d16',
     '30d4954d-a288-4bb3-8686-848fcf779e72',
     '755f897e-15ba-4e99-89f6-07332ef5b9d3',
     '088695e4-8624-405c-b33c-548ec48d7fdb',
     'a82bbf6e-0982-45f0-b63b-3e221b8bf473',
     'ba6795c8-1d5e-4d01-8ff9-ccd5b926a588',
     '19017d35-b2c1-4bef-9f93-d73e0342da4f',
     '7c3ce3b6-4bc9-4b89-b340-42dc93b673dd',
     '6f8ace61-7d48-441a-80f1-a0de28c6640c',
     '84b602c6-252b-44e7-964d-1cbe06e5ea85',
     '421e3407-8136-4131-a255-a4bd189cb92f',
     '667f325c-12ee-4b85-807c-fe69fa3d92bb',
     'd6ac90ed-560b-4f23-9052-d7890cf43963',
     '5b4cbf31-45a2-4b7c-8098-be1d74c10b33',
     'a60b1a1b-a109-4a25-9b12-30b2bc54c47a',
     'ef11d019-c871-453b-9209-1574810b94e2',
     '99d7589c-83a6-4b51-8191-30323f8f544c',
     'b70a930f-c8a1-4c6a-ac66-5451cc8f8818',
     'a41d13e7-0e8b-402c-be3c-26313c7d28ce',
     '0e315962-e85b-4807-a2a0-7655c6316146',
     'b607db1a-f141-44cd-b1c9-6374aa2d2ca0',
     '98383f31-19be-4f13-b5a0-21ab5f90319f',
     '70079841-4c72-4ba6-815a-25865a36611f',
     '2922683e-b860-4d53-b068-a46fe23dc0c1',
     'c0cb2f98-4801-4f44-ae5a-437023b36c9b',
     '76941479-bd6c-4398-a251-7296cc88de66',
     'fd3993c4-9c4d-44ec-a266-00e142403b27',
     'ab375b3d-ef49-4a55-af6c-92b587fda293',
     '8f19100e-1f4f-4edb-97f2-3ae3ce79e6cb',
     '646acd2f-5a8b-4629-b753-faff3ff49b8c',
     'cd045db2-56f9-4c52-9ae4-7d856c89396c',
     'b9f0362c-c11b-412c-8e94-d233efac474c',
     '3e246b13-acc5-4a6c-a0dc-a34560ce332e',
     'af75cd86-8186-4cd1-b758-f26e167803a0',
     'c0ae4eda-7838-48f7-bafb-9c7a56992453',
     'cf88c477-17c9-42dd-ab02-da163b898a26',
     'e46683e2-02d7-411b-a811-8e58a32907c6',
     'ca8e5a66-faa7-4f07-8773-7493d1fb73f2',
     'b35f2df2-c579-483d-af38-4af0b7e4d515',
     'b0d47d31-c019-4b62-b7ad-98228ddaaa67',
     '6b625968-05ff-4832-940b-fb67fef081f5',
     '65a19fd5-102f-4aae-8191-f49d96b3a530',
     '2cbfe131-4d22-4fc3-b0f4-0c8ef4395d7b',
     '6667aef8-ac61-4a6c-9a0b-8435a77795e1',
     'b5018344-94aa-47c1-931e-0bbe779d10c8',
     '83df1b22-c83a-4412-8e27-1c113bd17907',
     '1eb1aeb9-04a2-48d7-bdf7-ea73cbbf53f5',
     'de0b62ec-112f-45d0-a179-12425c433a34',
     'e57ac1f3-01bc-4b08-a599-ef11640730ac',
     'f2a07470-550d-4584-a72f-95643e0d8805',
     '6fc6f17a-de23-4dfb-b6cc-7cf223def0a0',
     'bef7787b-7047-4f5b-891d-6dbf1ab520ba',
     'bedec544-8c16-47b7-9aa8-7d1de56e5603',
     '215a6005-bb4b-4f11-83e4-74aa016104ac',
     '4cda34a9-7321-406b-9846-c54194de1493',
     '9c66d416-e194-4faf-8c63-8ca4e28aefff',
     'c326fe66-ba20-4f8a-ac61-aa29258ea4c6',
     '4bba8ed7-c2ef-4c14-bf3e-002ecb658f38',
     '6ac33fb4-6411-4e4e-b1bf-8d778b006a4f',
     '08706d0e-b702-4d98-87a3-1e74641e1751',
     '89621e8c-e224-43a8-9e91-e173ab92b473',
     '199306d2-0bfb-4531-bc01-63c7be0ee9b5',
     'deea2192-d298-4ecf-84f8-6262290c0f3b',
     '11752721-25a2-4881-995f-cd319c9451c7',
     'c13971b0-f24d-4d53-acc7-3d7084039746',
     'eabf04cd-d09a-4c6f-8120-27d185e14f01',
     'b155ac53-aa94-4404-9f2d-858432a37f0e',
     'c3a0a923-1861-45f5-996e-a504abc9769a',
     'fecf18d0-a58a-4d11-a353-ea73fbfa6860',
     '100961aa-70bb-4de4-acd1-223d23c9eb80',
     '53c957b3-2a8e-4ea9-973c-534b5e540b74',
     '6088ffb1-1de1-41a8-bb04-65eb28535a37',
     '5f99c308-40e4-480b-8448-273f8eb9aec2',
     'e3802510-e8b2-4e37-a5aa-a2df060e44e6',
     '87086cdb-52da-4425-bda4-e772afa32882',
     '8476c87b-dc78-41cc-beb0-0d278e230671',
     'c0a54432-3243-455e-8daa-7d59680ef098',
     'cfd0933b-4b9f-4daf-8526-75e4ddee1dbb',
     'f578383b-66e8-4491-8182-43bd1971756c',
     'f925cd19-3744-4bce-8aba-27b77747d33b',
     '28c3f3ed-c351-4ca1-ab48-db690f7c0faa',
     '91d61ecd-fc5a-4f0e-94b3-7139e23c53f9',
     'b9dd1ced-4c2f-4d0a-a4f1-ee5c5712c807',
     'e0341b71-28a9-448e-b0a7-0e8efbbdad65',
     '4a31f6e9-8f0a-4032-bc80-81192396eb65',
     '843dd9e2-98fd-4fce-9ac2-bcbe7ef7ee88',
     'adeee4ed-5118-4ffa-8ba6-610af6abedc4',
     '512c61af-7f5f-45ca-a098-1c5497c39e41',
     '41191966-1987-45d8-8bf8-dd0952b88ff0',
     '89358acb-5da6-40d0-8c55-5c72dbc12338',
     '61fd0a4d-35bc-41a3-978b-24dce7b5a61b',
     '3cc14c3e-8586-4017-80c1-152954ebbe2a',
     '6d69efb9-68bf-4051-8cdc-c7d72af9f419',
     '19b2917a-72dc-4b25-ade1-eae2c2c7c036',
     'bfc98bf9-8728-4360-9197-5c82d49f3c4a',
     '15386729-e035-4922-b941-2bb8b3eb70f0',
     'b05d01ec-ab23-4133-aa6a-6420f8333270',
     '79723382-af35-4dd2-8ccc-b71ee1fc578f',
     '9bc5dcd2-8f6b-4d70-918e-ad5d0f795ce6',
     'd707fa53-a45e-41a4-921f-2462c4d4c011',
     '17d3e2a1-84d9-48c5-9dde-af9f51f6b358',
     'ecb8924f-931a-4e82-93ea-e6c740a20212',
     '26f84f1c-4c2e-43ed-8438-d7a9d4bd58e4',
     '42b2f919-79ef-4788-9e5d-3120bf01cc6a',
     '51bc1d92-444a-454b-bc76-77c289064d20',
     '7a2cff82-cc76-44d0-bba7-0c8952982990',
     'eccad7b2-f533-4479-8342-c457275e39c8',
     '06b816c0-a9fe-43ab-9bf0-670a447aea1e',
     '6ee99712-b3ae-4b6e-9728-5577bfda54df',
     'f9f75dc7-9293-4588-aef1-c6dae84bee01',
     '1d239009-b497-413e-b2fe-c1f3df618ab6',
     'e4ed043b-e399-42e9-8ff7-5415ca1ae7dd',
     'd607faa5-11b5-4ea6-bd6b-03715d8bd6cc',
     'af1c9d56-d614-4a31-bff8-36f57a3b0330',
     '98d27851-3ec8-417a-91f6-ddb73152d048',
     'ba499004-ccbf-42ce-92e9-3c8c38f088d8',
     '2c7fceaf-a902-4900-8e76-5859fd99148b',
     '2e444aad-2394-4164-ab1f-14c5e537dd49',
     '02a9a8b4-e2f3-4b53-a246-78dd02bb2d03',
     '941ca60f-18fc-4321-a257-c6a656e1b5ec',
     '08929fc3-2331-4c9a-bd15-68147ea87d01',
     'e6240556-2492-4c2f-95aa-0571f6268717',
     '1cb385e6-8d9f-4bd2-9273-81c16e5d67d7',
     '9d1c473d-7df0-4f60-9509-ab6b2027b72a',
     '8285fc1e-5f9f-41c7-a232-8b1dc28d3bf3',
     'a52c1e17-97fa-4339-a21e-721945894c90',
     'faa80c7f-4baa-4a70-abb0-867ad4d78138',
     'e1a48fbb-01ec-408a-8135-680b21c9f0bf',
     '8fdecfaa-9711-4671-823b-34f717232849',
     '887768de-fc18-4445-98d3-a33ee2b937c5',
     '663f264a-cf9d-47cd-83c9-866f5a2c3fae',
     '51bc423f-c83c-4a50-a7e7-49fa74b14e35',
     '533b6bfb-161e-4f88-b021-f10efb3ffe5a',
     '347ce4d3-7eb0-4eae-8900-1065a95ed1c6',
     '87256cb8-ac47-4b25-8053-8e33b66f7b3d',
     '7d7a5092-b244-44a2-a776-3bb2c8d5a80f',
     '8c6b71b8-1e56-4cac-ae44-ea2dc0cbfc71',
     '4e909149-73d7-45fa-b06d-308b6ca323ca',
     '059a058b-784a-4a5e-8688-11ee7c2dd5e7',
     'f764b54b-57e9-4751-9d9e-3d513fdb0202',
     '7ff692df-ca89-4a17-ace1-674cb0e7afb8',
     '12e97936-b9be-4df4-abb5-021d26168d34',
     '68e7e365-19e4-4645-aa01-c605ffe1ff1b',
     'cba71e6c-fbe0-4ee8-9fc6-b68525a2168b',
     'fe9b193e-d82e-4612-97de-051cb7d87c2c',
     '9bcc366c-bbda-48a7-866d-83039416a1c1',
     'a6ed52e9-afd3-4ba3-8b37-952654081888',
     'acc341c3-353b-4b36-b0ef-785e3f002bfc',
     'b0532a5b-073a-4b6a-9a21-4d141a86b5f1',
     'b3a73dea-1cd4-4fd7-8bf5-ff9ccbd9ef17',
     'c5fd17e0-be5e-411a-ae90-6da111c0686b',
     '249ad276-075d-4dba-81a2-c375860e29ce',
     'ac0357e8-e6e3-4058-8841-54e0d88942eb',
     '9b2ec640-1e46-4b66-bbc4-eb997fd04dd9',
     '2e145490-dbe0-4d1f-956f-121b2473afbc',
     '5a841d94-fdce-435b-a77a-3105bfc2c8c9',
     '3fc49b3f-95e1-4874-8ffe-df14e6874d46',
     'd9ee2185-5120-40ef-a4b0-2778178d8fd1',
     '1603d1af-1680-4c9c-a915-b47dbb16331f',
     '8c196e62-1ddb-4162-8a82-8225e877f078',
     'ddc1b989-8c2f-426f-b3af-892434f066d7',
     '80842557-52ff-482c-8c45-f3d7afffe326',
     '6d63b87e-47bf-4499-8503-b608fd2592c1',
     '3b858662-1220-4f03-86d7-aba3efdf6973',
     'a4a1e766-a37b-4848-9193-911a66a34de8',
     'f77e9e18-7090-4e4a-af77-114505b29e97',
     '59ff1592-05d9-44ac-913b-f76c41786a9f',
     '4afc1bc7-4351-4cfc-80ff-a9c8264b6032',
     'd0a72577-8917-444a-a6f1-c8f507cd0bde',
     '14355a7d-6b97-4a5f-9564-75472a126c56',
     '40cb7bac-9983-4640-bf6b-0b8668288aba',
     'ef8e47cc-604d-46fd-b33b-19f59e3dcb01',
     '0096491a-77bf-4e79-b906-fc7a8e5e57a6',
     '21021cae-d56f-4842-8cb3-9bf2c6ee5605',
     'f0475494-14ba-4ecb-8ee5-ec084fb7a191',
     'c2df6a43-9caf-4e41-9ed5-690e15bde140',
     '1da62d3c-ed83-4847-b334-7b8708bee20c',
     'ad660174-aa90-47a2-b4b3-d19cb242fe2a',
     'edf4f0cb-c9d0-4832-a718-d3e2acc461f1',
     '4ca965ca-1794-4f5f-a14a-4a0ba41137d3',
     'ed48a75f-d0aa-46d9-9688-207fa2fa5039',
     '882f55fd-273b-4df7-bd91-4eb3a246d0ab',
     '2580dd08-e94c-43d1-a429-2e0d48533754',
     'bcafdba4-ee4b-4df9-bbb5-64d2d8267011',
     '7de2d2cb-7745-496b-a64d-caf5c42df673',
     'a72e6ba6-87ad-4e25-a6b9-d61ee868b8ca',
     '46c04283-6397-46ee-865a-bfe125a5a8ab',
     'd605183a-9ffe-41ae-ab27-d76dbb3d8a9d',
     '8c07aa1d-65c2-4696-baf3-6ac7e0e1a0c9',
     'bfcfcaa9-2f28-4ec4-8bee-db2116d178cc',
     'd1221601-eb19-4a6c-9e97-e3ddcf960386',
     '7c6df630-104f-4a96-a94f-1f5a177db42f',
     '1d9eb5b0-9ed8-4663-8c89-d611abb4764a',
     'ab915c05-cd34-40bf-b4fd-321ac00304ec',
     '062f021e-8297-41e6-a8fe-54a382ab7404',
     '23c2a356-c842-41ee-8cbc-2efa72731be0',
     'dcd473ef-d1f6-4b05-a2c3-b03f71bc8c33',
     'aa2a8eb5-4667-4da8-a64f-866a71080973',
     'ff55bab4-812b-405b-9856-8bedf9fb11f9',
     'a06314f1-64f4-433d-984c-c0a614187ca1',
     'c7ebd263-85a4-48e7-83f5-80a12bd309f0',
     '4a748280-132e-4c66-8ec6-c5ed052c3f8d',
     '107fff84-6401-493b-8b3e-d299622ba1db',
     'de8a6608-2b0d-410e-bb68-66a845f46c89',
     '53f3a54c-37ba-49c0-a9eb-a9b1734c1853',
     'd76083a0-6c0d-446c-82e3-723a6d27c098',
     'd3de98b6-a62c-44b8-9351-f074e4cdb106',
     'b0966f14-dab9-4592-9d64-81bb9a34d0ea',
     '91dc4688-8549-446f-b7a5-06306b454425',
     '4cfd0c06-1f36-4932-9336-6adaa36e4837',
     'a72e0c37-7b10-4d93-846c-0055392b4835',
     '823b7a4d-4a72-468d-b7c8-79ebae3ad901',
     '7768eca6-5712-4260-a419-b34cc3466988',
     'f03fcac7-b005-4ba0-952e-b8616cbaa093',
     '72419be6-b3b7-4ceb-8361-ad1cc116eb98',
     '5e8f7cca-fa0b-4a26-b415-b40a47d4fe7d',
     '340bc7d4-1118-4e5f-acd3-97312a2b1425',
     'bf831286-0b84-4dd6-9256-6684659d9fa7',
     'fb209eba-f92e-4669-aa27-e0fecd43df1d',
     'e553045c-d9e7-4209-b8d5-62ce577b29d7',
     'e1312b24-da88-4668-bea1-16d69e431d4f',
     '3cc37f5f-192b-4e11-b888-7745d7406efd',
     '6ccfb911-f5cb-4026-a8b5-46174a2aa441',
     '9ca7abb1-91ca-4070-9fbd-f41cb954c732',
     'cc79d3fc-ccc0-48a2-8621-76b70b37ca0f',
     '58ed8ca5-3645-462e-b585-e7b1f667f7b5',
     '46570f69-6a73-4f3d-ab89-aa1f5145a2c7',
     'fa027255-7e4d-4b79-91bf-a2a7b9f286a5',
     '9e23bebc-0bca-4b76-b9a8-2f4375d802de',
     '7c2531ff-3461-42c6-985c-51d395dee69a',
     '87e900ef-2b38-4f28-b440-100eccf6b449',
     'c9e3518a-04e5-4159-be32-49dba9137ed8',
     '5dd6ec46-260c-47e7-81ed-5fbc626f2eb6',
     'c5889045-7b70-48aa-839f-873234b4af3d',
     'ceb505f9-7986-421f-b18c-bb19009aaf2f',
     'b9b6f8cf-f3e7-4db3-9158-c2f492da93e1',
     '2cffa44a-cff4-40cc-9059-84792d4497b0',
     '232d4dac-2f36-44c8-877e-51fe078b6bf7',
     'deb2f50b-fdc8-48f2-9795-248efffd658e',
     '6d1bc5ff-1464-4e5e-adee-f1b81f875ffb',
     '09f6d521-d08e-4d79-af38-b5363a5fcddb',
     '74863d5d-b75d-497c-bdf3-45335d31a48c',
     '1132ab6f-8be6-441d-9be9-6b2eb0c1efe9',
     'e4487b9b-5183-4aaa-8044-6b091cfb48bd',
     '56739035-84ab-4ee6-9fb9-4fb45477d591',
     'ed97c30e-d078-4c7f-8171-9944801e2e88',
     '495dd320-61a3-456a-bb26-f316b8508d42',
     'b65de460-03f9-4b2b-add2-658ac12a00fb',
     '188d45ee-832d-40a6-bd57-05e97d88e631',
     'afdb7961-b300-4344-8e0f-3d94d63ac900',
     '41a076f2-1b8f-4eec-ada6-52cffac899a4',
     '76fe6c73-fc36-4e8f-bfcb-dfac793ca22e',
     'fcdfb64b-0dd7-4839-a76d-1bd3d688d5c0',
     'c4547da8-3466-4d63-9102-fa2d42ce28dd',
     'ceb537f7-5bae-475e-8fa0-0806abaaf831',
     'f686841b-8ead-475a-855e-444cbe449812',
     '712f9962-fc6c-404a-b7cf-bc0dc69ce915',
     ...]




```python
# Get unique student IDs
unique_student_ids = page_views_df['student_id'].unique()

# Indegree Calculation
indegree_sums = {} 

for student_id in unique_student_ids:
    # Filter data for the current student
    student_data = page_views_df[page_views_df['student_id'] == student_id]
    student_data = student_data.sort_values(by='dt_accessed') 

    # Create a graph
    G = nx.DiGraph()
    G.add_nodes_from(student_data['chapter_section'].unique())
    for i in range(len(student_data) - 1):
        source = student_data.iloc[i]['chapter_section']
        target = student_data.iloc[i + 1]['chapter_section']
        G.add_edge(source, target)

    # Calculate indegree for each node
    for node in G.nodes():
        indegree_sums.setdefault(node, 0)  
        indegree_sums[node] += G.in_degree(node)

# Outdegree Calculation
outdegree_sums = {} 

for student_id in unique_student_ids:
    # Filter data for the current student
    student_data = page_views_df[page_views_df['student_id'] == student_id]
    student_data = student_data.sort_values(by='dt_accessed') 

    # Create a NEW graph for each student
    G = nx.DiGraph()
    G.add_nodes_from(student_data['chapter_section'].unique())
    for i in range(len(student_data) - 1):
        source = student_data.iloc[i]['chapter_section']
        target = student_data.iloc[i + 1]['chapter_section']
        G.add_edge(source, target)

    # Calculate outdegree for each node in the current student's graph
    for node in G.nodes():
        outdegree_sums.setdefault(node, 0)  
        outdegree_sums[node] += G.out_degree(node)
```


```python
# Extract nodes (chapter sections) and their indegrees
nodes = list(indegree_sums.keys())
indegree_values = list(indegree_sums.values())

# Create a bar chart
plt.figure(figsize=(24, 24))
plt.bar(nodes, indegree_values)

# Customize appearance
plt.xlabel("Chapter Section")
plt.ylabel("Indegree")
plt.title("Indegree Distribution of Chapter Sections")
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
```


    
![png](output_59_0.png)
    



```python
# Visualize Outdegrees
nodes = list(outdegree_sums.keys())
outdegree_values = list(outdegree_sums.values())

plt.figure(figsize=(24, 24))
plt.bar(nodes, outdegree_values)
plt.xlabel("Chapter Section")
plt.ylabel("Outdegree")
plt.title("Outdegree Distribution of Chapter Sections")
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()
```


    
![png](output_60_0.png)
    



```python
# Calculate total indegree and outdegree
total_indegree = sum(indegree_sums.values())
total_outdegree = sum(outdegree_sums.values())

# Print results
print("Total Indegree:", total_indegree)
print("Total Outdegree:", total_outdegree)

# Check if they are equal
if total_indegree == total_outdegree:
    print("The sum of indegree and outdegree are equal!")
else:
    print("Warning: The sum of indegree and outdegree are not equal.")
```

    Total Indegree: 296119
    Total Outdegree: 296119
    The sum of indegree and outdegree are equal!
    


```python
# Calculate indegree - outdegree
net_degree = {}
for node in indegree_sums:
    net_degree[node] = indegree_sums[node] - outdegree_sums.get(node, 0)

# Extract nodes and net degree values
nodes = list(net_degree.keys())
net_degree_values = list(net_degree.values())

# Create the bar chart
plt.figure(figsize=(18, 8))  # Adjust figure size as needed 
plt.bar(nodes, net_degree_values)

# Customize appearance
plt.xlabel("Chapter Section")
plt.ylabel("Indegree - Outdegree")
plt.title("Net Degree Distribution of Chapter Sections")
plt.xticks(rotation=90, ha='right')
plt.tight_layout()

# Assuming you have a list of indices where chapters begin:
chapter_start_indices = [1, 6, 17, 28, 44, 56, 70, 81, 89, 102, 109, 117, 123, 132, 143, 151] 

# Add vertical lines
for index in chapter_start_indices:
    plt.axvline(x=index - 0.5, color='gray', linestyle='--')

plt.show()
```


    
![png](output_62_0.png)
    



```python
# Function to encode chapter sections
def encode_chapter_section(section):
    if pd.isna(section):
        return np.nan
    try:
        major, minor = map(float, section.split('_'))
        return major + minor / 100  # Correct scaling for the minor part
    except:
        return np.nan

# Adjusted function for calculating edge types, to work with grouped DataFrame
def calculate_edge_types(group):
    num_forward_edges = 0
    num_backward_edges = 0
    
    for i in range(len(group) - 1):
        current_chapter = group.iloc[i]['chapter_section']
        next_chapter = group.iloc[i + 1]['chapter_section']

        if pd.isna(current_chapter) or pd.isna(next_chapter):
            continue

        if next_chapter < current_chapter:
            num_backward_edges += 1
        else:
            num_forward_edges += 1

    # Return a Series with calculated values
    return pd.Series({'forward_edges': num_forward_edges, 'backward_edges': num_backward_edges})

# Apply encoding to 'chapter_section'
page_views_df['chapter_section'] = page_views_df['chapter_section'].apply(encode_chapter_section)

# Drop rows with NaN in 'chapter_section'
page_views_df.dropna(subset=['chapter_section'], inplace=True)

# Group by 'student_id' and apply calculation, do not reset index to keep 'student_id' as index
edges_df = page_views_df.groupby('student_id').apply(calculate_edge_types)
```

    C:\Users\btrok\AppData\Local\Temp\ipykernel_44956\108616040.py:38: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      edges_df = page_views_df.groupby('student_id').apply(calculate_edge_types)
    


```python
page_views_df = pd.merge(edges_df, page_views_df, on='student_id', how='inner')
```


```python
page_views_df['edge_ratio'] = page_views_df['backward_edges'] / (page_views_df['backward_edges'] + page_views_df['forward_edges'])
```


```python
# Assuming page_views_df is your DataFrame
# Remove rows with NaN values in either 'edge_ratio' or 'mean_accuracy'
page_views_df_clean = page_views_df.dropna(subset=['edge_ratio', 'mean_accuracy'])

# Define predictor (X) and response (y) variables
X = page_views_df_clean['edge_ratio']
y = page_views_df_clean['mean_accuracy']

# Add a constant to the predictor variable array for the intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>mean_accuracy</td>  <th>  R-squared:         </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   35.68</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 27 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>2.33e-09</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>20:09:23</td>     <th>  Log-Likelihood:    </th> <td>1.2943e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>467506</td>      <th>  AIC:               </th> <td>-2.589e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>467504</td>      <th>  BIC:               </th> <td>-2.588e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    0.6421</td> <td>    0.001</td> <td>  775.162</td> <td> 0.000</td> <td>    0.641</td> <td>    0.644</td>
</tr>
<tr>
  <th>edge_ratio</th> <td>    0.0243</td> <td>    0.004</td> <td>    5.973</td> <td> 0.000</td> <td>    0.016</td> <td>    0.032</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21804.266</td> <th>  Durbin-Watson:     </th> <td>   0.083</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>20970.250</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.471</td>   <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.566</td>   <th>  Cond. No.          </th> <td>    15.7</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Simulated data (replace with your actual data)
data = {
    'edge_ratio': X['edge_ratio'],  # Assuming X is your DataFrame of features
    'mean_accuracy': y,  # Actual values
    'predicted_accuracy': Y_pred  # Model predictions
}
df = pd.DataFrame(data)
df = df.drop_duplicates()
```


```python
# Set the aesthetic style of the plots
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# Create the plot
plt.figure(figsize=(18, 8))

ax = sns.regplot(x='edge_ratio', y='mean_accuracy', data=df, color='#102747', 
                 line_kws={'color': 'red'}, scatter_kws={'alpha':0.1})

ax.set_facecolor('#f0f0f0')

# Labeling the axes and title
plt.xlabel('Edge Ratio', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.title('Edge Ratio vs Mean Accuracy with Linear Regression', fontsize=16)

# Adjusting limits if necessary
plt.xlim([df['edge_ratio'].min() * 0.9, df['edge_ratio'].max() * 1.1])
plt.ylim([df['mean_accuracy'].min() * 0.9, df['mean_accuracy'].max() * 1.1])

plt.tight_layout()
plt.show()
```


    
![png](output_68_0.png)
    



```python
# Function to count loops within each chapter section
def count_loops_per_chapter(df):
    # Initialize a dictionary to hold the count of loops per chapter section
    loops_per_chapter = {}
    
    for i in range(len(df) - 1):
        current_chapter = df.iloc[i]['chapter_section']
        next_chapter = df.iloc[i + 1]['chapter_section']
        
        # Check if the current and next chapter sections are the same
        if current_chapter == next_chapter:
            # Increment the loop count for this chapter section
            if current_chapter in loops_per_chapter:
                loops_per_chapter[current_chapter] += 1
            else:
                loops_per_chapter[current_chapter] = 1
                
    return loops_per_chapter

# Apply the function to the entire dataframe to get the count of loops per chapter
loops_df = count_loops_per_chapter(page_views_df)

# Convert the dictionary to a DataFrame for easier handling
loops_df = pd.DataFrame(list(loops_df.items()), columns=['chapter_section', 'loops'])
```


```python
def visualize_loops_sorted_by_chapter(loops_df):
    # Ensure 'chapter_section' is in float format to sort numerically
    loops_df['chapter_section'] = loops_df['chapter_section'].astype(float)
    
    # Sort the DataFrame by 'chapter_section' numerically
    loops_df_sorted = loops_df.sort_values('chapter_section')
    
    plt.figure(figsize=(18, 18))
    plt.bar(loops_df_sorted['chapter_section'].astype(str), loops_df_sorted['loops'], color='skyblue')
    plt.xlabel('Chapter Section')
    plt.ylabel('Number of Loops')
    plt.title('Number of Loops per Chapter Section')
    
    # Rotate the x labels to make them more readable if there are many sections
    plt.xticks(rotation=90, ha="right")
    
    plt.tight_layout()  # Adjust layout to make room for the rotated x-labels
    plt.show()

# Call the visualization function with the DataFrame containing loop counts
visualize_loops_sorted_by_chapter(loops_df)
```


    
![png](output_70_0.png)
    



```python
# Visualization
def visualize_loops(loops_df):
    # Sorting the DataFrame based on 'loops' to have a meaningful chart
    loops_df_sorted = loops_df.sort_values('loops', ascending=False)
    
    plt.figure(figsize=(18, 18))
    plt.bar(loops_df_sorted['chapter_section'].astype(str), loops_df_sorted['loops'])
    plt.xlabel('Chapter Section')
    plt.ylabel('Number of Loops')
    plt.title('Number of Loops per Chapter Section')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
visualize_loops(loops_df)
```


    <Figure size 640x480 with 0 Axes>



    
![png](output_71_1.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example calculation for total page views per chapter section in page_views_df
page_views_summary = page_views_df.groupby('chapter_section').size().reset_index(name='page_views')

# Merging loops_df with the page views summary to calculate the ratio
loops_df = pd.merge(loops_df, page_views_summary, on='chapter_section', how='left')
loops_df['loops_per_view'] = loops_df['loops'] / loops_df['page_views']

# Sorting based on 'loops_per_view' for a meaningful chart
loops_df_sorted = loops_df.sort_values('loops_per_view', ascending=False)

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plotting with improvements
plt.figure(figsize=(18, 10))  # Adjusted for better fit
plt.bar(loops_df_sorted['chapter_section'].astype(str), loops_df_sorted['loops_per_view'], color='#102747')

plt.xlabel('Chapter Section', fontsize=14)
plt.ylabel('Loops per Page View', fontsize=14)
plt.title('Loops per Page View for Each Chapter Section', fontsize=16)

# Customizing x-axis labels for better readability
plt.xticks(rotation=90, ha='right')

# Setting the background color and gridlines for aesthetics
plt.gca().set_facecolor('#f0f0f0')
plt.grid(color='white', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
```


    
![png](output_72_0.png)
    



```python
page_views_df['page'].unique()
```




    array(['1.3 Doing Statistics with R', '1.4 Introduction to R Functions',
           '1.5 Save Your Work in R Objects', '1.6 Goals of This Course',
           '1.7 Chapter 1 Review Questions',
           '2.1 Starting with a Bunch of Numbers', '2.2 From Numbers to Data',
           '2.3 A Data Frame Example: MindsetMatters', '2.4 Measurement',
           '2.5 Measurement (Continued)', '2.6 Sampling from a Population',
           '2.7 The Structure of Data', '2.8 Manipulating Data',
           '2.9 Summary', '2.10 Chapter 2 Review Questions',
           '2.11 Chapter 2 Review Questions 2',
           '3.1 The Concept of Distribution',
           '3.2 Visualizing Distributions with Histograms',
           '3.3 Shape, Center, Spread, and Weird Things',
           '3.4 The Data Generating Process',
           '3.5 The Back and Forth Between Data and the DGP',
           '3.6 The Back and Forth Between Data and the DGP (Continued)',
           '3.7 The Five-Number Summary',
           '3.8 Boxplots and the Five-Number Summary',
           '3.9 Exploring Variation in Categorical Variables',
           '3.10 Chapter 3 Review Questions',
           '3.11 Chapter 3 Review Questions 2',
           '4.1 Welcome to Explaining Variation',
           '4.2 Explaining One Variable with Another',
           '4.3 Outcome and Explanatory Variables',
           '4.4 More Ways to Visualize Relationships: Point and Jitter Plots',
           '4.5 Even More Ways: Putting these Plots Together',
           '4.6 Representing Relationships Among Variables',
           '4.7 Sources of Variation', '4.8 Randomness',
           '4.9 From Categorical to Quantitative Explanatory Variables',
           '4.10 Quantitative Explanatory Variables', '4.11 Research Design',
           '4.12 Considering Randomness as a Possible DGP',
           '4.13 Shuffling Can Help Us Understand Real Data Better',
           '4.14 Quantifying the Data Generating Process',
           '4.15 Chapter 4 Review Questions',
           '4.16 Chapter 4 Review Questions 2',
           '5.1 What is a Model, and Why Would We Want One?',
           '5.2 Modeling a Distribution as a Single Number',
           '5.3 Median vs. Mean as a Model', '5.4 Exploring the Mean',
           '5.5 Fitting the Empty Model',
           '5.6 Generating Predictions from the Empty Model',
           '5.7 Thinking About Error',
           '5.8 The World of Mathematical Notation',
           '5.9 DATA = MODEL + ERROR: Notation',
           '5.10 Summarizing Where We Are', '5.11 Chapter 5 Review Questions',
           '5.12 Chapter 5 Review Questions 2',
           '6.1 Quantifying Total Error Around a Model',
           '6.2 The Beauty of Sum of Squares', '6.3 Variance',
           '6.4 Standard Deviation', '6.5 Z-Scores',
           '6.6 Interpreting and Using Z-Scores',
           '6.7 Modeling the Shape of the Error Distribution',
           '6.8 Modeling Error with the Normal Distribution',
           '6.9 Using the Normal Model to Make Predictions',
           '6.10 Getting Familiar with the Normal Distribution',
           '6.11 The Empirical Rule', '6.12 Next Up: Explaining Error',
           '6.13 Chapter 6 Review Questions',
           '6.14 Chapter 6 Review Questions 2', '7.1 Explaining Variation',
           '7.2 Using R to Fit the Group Model',
           '7.3 GLM Notation for the Group Model',
           '7.4 How the Model Makes Predictions',
           '7.5 Error Leftover From the Group Model',
           '7.6 Graphing Residuals From the Model',
           '7.7 Error Reduced by the Group Model',
           '7.8 Using SS Error to Compare Group to Empty Model',
           '7.9 Partitioning Sums of Squares into Model and Error',
           '7.10 Using Proportional Reduction in Error (PRE) to Compare Two Models',
           '7.11 Chapter 7 Review Questions',
           '8.1 Extending to a Three-Group Model',
           '8.2 Fitting and Interpreting the Three-Group Model',
           '8.3 Comparing the Fit of the Two- and Three-Group Models',
           '8.4 The F Ratio', '8.5 Modeling the DGP',
           '8.6 Using Shuffle to Compare Models of the DGP',
           '8.7 Measures of Effect Size', '8.8 Chapter 8 Review Questions',
           '9.1 Using a Quantitative Explanatory Variable in a Model',
           '9.2 Specifying the Height Model with GLM Notation',
           '9.3 Interpreting the Parameter Estimates for a Regression Model',
           '9.4 Comparing Regression Models to Group Models',
           '9.5 Error from the Height Model',
           '9.6 Sums of Squares in the ANOVA Table',
           '9.7 Assessing Model Fit with PRE and F', '9.8 Correlation',
           "9.9 More on Pearson's R",
           '9.10 Using Shuffle to Interpret the Slope of a Regression Line',
           '9.11 Limitations to Keep in Mind',
           '9.12 Chapter 9 Review Questions',
           '9.13 Chapter 9 Review Questions 2',
           '10.1 The Problem of Inference',
           '10.2 Constructing a Sampling Distribution',
           '10.3 Exploring the Sampling Distribution of b1',
           '10.4 The p-Value',
           '10.5 A Mathematical Model of the Sampling Distribution of b1',
           '10.6 Things That Affect p-Value',
           '10.7 Hypothesis Testing for Regression Models',
           '11.1 Moving Beyond b1',
           '11.2 Sampling Distributions of PRE and F',
           '11.3 The Sampling Distribution of F',
           '11.4 The F-Distribution: A Mathematical Model of the Sampling Distribution of F',
           '11.5 Using F to Test a Regression Model',
           '11.6 Type I and Type II Error',
           '11.7 Using F to Compare Multiple Groups',
           '11.8 Pairwise Comparisons',
           '12.1 From Hypothesis Testing to Confidence Intervals',
           '12.2 Using Bootstrapping to Calculate the 95% Confidence Interval',
           '12.3 Shuffle, Resample, and Standard Error',
           '12.4 Interpreting the Confidence Interval',
           '12.5 Confidence Intervals for Other Parameters',
           '12.6 What Affects the Width of the Confidence Interval',
           '13.1 Models with Two Explanatory Variables',
           '13.2 Visualizing Price = Home Size + Neighborhood',
           '13.3 Specifying and Fitting a Multivariate Model',
           '13.4 Interpreting the Parameter Estimates for a Multivariate Model',
           '13.5 Predictions from the Multivariate Model',
           '13.6 Using Residuals and Sums of Squares to Measure Error Around the Multivariate Model',
           '13.7 Using Venn Diagrams to Conceptualize Sums of Squares, PRE, and F',
           '13.8 The Logic of Inference with the Multivariate Model',
           '13.9 Using the Sampling Distribution of F',
           '14.1 Targeted Model Comparisons',
           '14.2 Sums of Squares for Targeted Model Comparisons',
           '14.3 PRE and F for Targeted Model Comparisons',
           '14.4 Inference for Targeted Model Comparisons',
           '14.5 Using `shuffle()` for Targeted Model Comparisons (Part 1)',
           '14.6 Using `shuffle()` for Targeted Model Comparisons (Part 2)',
           '14.7 Deciding Which Predictors to Include in a Model',
           '14.8 Models with Multiple Categorical Predictors',
           '14.9 Error and Inference from Models with Multiple Categorical Predictors',
           '14.10 Models with Multiple Quantitative Predictors',
           '14.11 Error and Inference from Models with Multiple Quantitative Predictors',
           '13.1 What You Have Learned About Exploring Variation',
           '13.2 What You Have Learned About Modeling Variation',
           '13.3 What You Have Learned About Evaluating Models',
           '8.5 Measures of Effect Size', '8.6 Modeling the DGP',
           '8.7 Using Shuffle to Compare Models of the DGP',
           '15.1 Dogs in the Emergency Room',
           '15.2 Additive versus Non-Additive Models',
           '15.3 Representing the Interaction Model in GLM Notation',
           '15.4 Interpreting Parameter Estimates for the Interaction Model',
           '15.5 Making Parameter Estimates More Meaningful in Interaction',
           '15.6 Centering a Quantitative Predictor at 0',
           '15.7 Comparing the Interaction Model to the Additive Model (Part 1)',
           '15.8 Comparing the Interaction Model to the Additive Model (Part 2)',
           '16.1 Interactions with Two Quantitative Predictors',
           '16.2 Fitting and Visualizing an Interaction Model with Two Quantitative Predictors',
           '16.3 Interpreting Parameter Estimates of Interaction Models with Two Quantitative Predictors',
           '16.4 Comparing the Interaction Model to the Additive Model with Two Quantitative Predictors',
           '16.5 Interactions with Two Categorical Predictors',
           '16.6 Predictions of the Interaction Model with Two Categorical Predictors',
           '16.7 Visually Comparing the Interaction Model to the Additive Model with Two Categorical Predictors',
           '16.8 Thinking of Factorial Models in Terms of Intercepts and Slopes'],
          dtype=object)




```python
# Take the top 20 rows
top20 = loops_df_sorted.head(20)

# Set the aesthetic style of the plots  
sns.set_style('whitegrid')

# Plotting with improvements
plt.figure(figsize=(18, 10)) 

plt.ylim([0.15, 0.35])

# Adjusted for better fit
plt.bar(top20['chapter_section'].astype(str), top20['loops_per_view'], color='#102747')
plt.xlabel('Chapter Section', fontsize=14)  
plt.ylabel('Loops per Page View', fontsize=14)
plt.title('Top 20 - Loops per Page View for Each Chapter Section', fontsize=16)

# Customizing x-axis labels for better readability  
plt.xticks(rotation=90, ha='right')

# Setting the background color and gridlines for aesthetics
plt.gca().set_facecolor('#f0f0f0')
plt.grid(color='white', linestyle='--', linewidth=0.5)  
plt.tight_layout()
plt.savefig("my_plot.png")
plt.show()
```


    
![png](output_74_0.png)
    



```python

```
