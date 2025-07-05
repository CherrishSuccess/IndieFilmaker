#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[15]:


# Unzip the file
with zipfile.ZipFile('TheMoviesDataset.zip', 'r') as zip_ref:
    zip_ref.extractall('movies_data')

# Check files
os.listdir('movies_data')


# In[25]:


# Load the main metadata file
df = pd.read_csv('movies_data/movies_metadata.csv', low_memory=False)
df.head()


# In[27]:


# Convert columns to numeric
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

# Filter out missing and invalid entries
df = df[(df['budget'] > 0) & (df['revenue'] > 0) & df['popularity'].notna()]

# Optional: log transformation
df['log_budget'] = np.log(df['budget'])
df['log_revenue'] = np.log(df['revenue'])


# In[30]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x='log_budget', y='log_revenue', data=df)
plt.title('Log(Budget) vs Log(Revenue)')
plt.xlabel('Log Budget')
plt.ylabel('Log Revenue')
plt.show()


# In[34]:


X = df[['log_budget', 'popularity']]
y = df['log_revenue']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# In[41]:


print("Intercept:", model.intercept_)

# Create a DataFrame of coefficients for clarity
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nCoefficients:\n", coefficients)


# In[39]:


y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)


# In[43]:


[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]


# In[45]:


import ast

def extract_primary_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list) and genres:
            return genres[0]['name']
        else:
            return np.nan
    except:
        return np.nan

df['primary_genre'] = df['genres'].apply(extract_primary_genre)
df['primary_genre'].value_counts().head()


# In[47]:


df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df = df[df['runtime'].notnull() & (df['runtime'] > 0)]
df = df[df['primary_genre'].notnull()]


# In[49]:


genre_dummies = pd.get_dummies(df['primary_genre'], prefix='genre', drop_first=True)
df = pd.concat([df, genre_dummies], axis=1)


# In[51]:


features = ['log_budget', 'popularity', 'runtime'] + list(genre_dummies.columns)
X = df[features]
y = df['log_revenue']


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)


# In[55]:


coefficients.set_index('Feature').plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importance in Predicting Log Revenue")
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.show()


# In[ ]:




