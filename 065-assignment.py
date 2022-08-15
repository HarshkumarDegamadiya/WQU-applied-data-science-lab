#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>6.5. Small Business Owners in the United States🇺🇸</strong></font>

# In this assignment, you're going to focus on business owners in the United States. You'll start by examining some demographic characteristics of the group, such as age, income category, and debt vs home value. Then you'll select high-variance features, and create a clustering model to divide small business owners into subgroups. Finally, you'll create some visualizations to highlight the differences between these subgroups. Good luck! 🍀

# In[48]:


import wqet_grader

wqet_grader.init("Project 6 Assessment")


# In[50]:


# Import libraries here
import pandas as pd
import plotly.express as px
import wqet_grader
from dash import Input, Output, dcc, html
from IPython.display import VimeoVideo
from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


JupyterDash.infer_jupyter_proxy_config()


# # Prepare Data

# ## Import

# Let's start by bringing our data into the assignment.

# **Task 6.5.1:** Read the file `"data/SCFP2019.csv.gz"` into the DataFrame `df`.

# In[51]:


df = pd.read_csv("data/SCFP2019.csv.gz")
print("df shape:", df.shape)
df.head()


# In[52]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.1", list(df.shape))


# ## Explore

# As mentioned at the start of this assignment, you're focusing on business owners. But what percentage of the respondents in `df` are business owners?

# **Task 6.5.2:** Calculate the proportion of respondents in `df` that are business owners, and assign the result to the variable `pct_biz_owners`. You'll need to review the documentation regarding the `"HBUS"` column to complete these tasks.

# In[53]:



pct_biz_owners = df["HBUS"].mean()

print("proportion of business owners in df:", pct_biz_owners)


# In[ ]:





# In[54]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.2", [pct_biz_owners])


# Is the distribution of income different for business owners and non-business owners?

# **Task 6.5.3:** Create a DataFrame `df_inccat` that shows the normalized frequency for income categories for business owners and non-business owners. Your final DataFrame should look something like this:
# 
# ```
#     HBUS   INCCAT  frequency
# 0      0     0-20   0.210348
# 1      0  21-39.9   0.198140
# ...
# 11     1     0-20   0.041188
# ```

# In[55]:


inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",
    5: "80-89.9",
    6: "90-100",
}

df_inccat = (
    df["INCCAT"].replace(inccat_dict)
    .groupby(df["HBUS"])
    .value_counts(normalize=True)
    .rename("frequency")
    .to_frame()
    .reset_index()
)

df_inccat


# In[ ]:



wqet_grader.grade("Project 6 Assessment", "Task 6.5.3", df_inccat)


# **Task 6.5.4:** Using seaborn, create a side-by-side bar chart of `df_inccat`. Set `hue` to `"HBUS"`, and make sure that the income categories are in the correct order along the x-axis. Label to the x-axis `"Income Category"`, the y-axis `"Frequency (%)"`, and use the title `"Income Distribution: Business Owners vs. Non-Business Owners"`.

# In[56]:


# Create bar chart of `df_inccat`
sns.barplot(x="INCCAT",y="frequency",hue="HBUS",data=df_inccat,order=inccat_dict.values())
plt.xlabel("Income Category")
plt.ylabel("Frequency (%)")
plt.title("Income Distribution: Business Owners vs. Non-Business Owners")
# Don't delete the code below 👇
plt.savefig("images/6-5-4.png", dpi=150)


# In[ ]:


with open("images/6-5-4.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.4", file)


# We looked at the relationship between home value and household debt in the context of the the credit fearful, but what about business owners? Are there notable differences between business owners and non-business owners?

# **Task 6.5.5:** Using seaborn, create a scatter plot that shows `"HOUSES"` vs. `"DEBT"`. You should color the datapoints according to business ownership. Be sure to label the x-axis `"Household Debt"`, the y-axis `"Home Value"`, and use the title `"Home Value vs. Household Debt"`. 

# In[ ]:


# Plot "HOUSES" vs "DEBT" with hue=label
sns.scatterplot(x="HOUSES",y="DEBT",data=df)
# Don't delete the code below 👇
plt.savefig("images/6-5-5.png", dpi=150)


# For the model building part of the assignment, you're going to focus on small business owners, defined as respondents who have a business and whose income does not exceed \\$500,000.

# In[ ]:


with open("images/6-5-5.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.5", file)


# **Task 6.5.6:** Create a new DataFrame `df_small_biz` that contains only business owners whose income is below \\$500,000.

# In[57]:


mask = ((df["HBUS"]==1) & (df["INCOME"]<500_000))
df_small_biz = df[mask]
print("df_small_biz shape:", df_small_biz.shape)
df_small_biz.head()


# In[ ]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.6", list(df_small_biz.shape))


# We saw that credit-fearful respondents were relatively young. Is the same true for small business owners?

# **Task 6.5.7:** Create a histogram from the `"AGE"` column in `df_small_biz` with 10 bins. Be sure to label the x-axis `"Age"`, the y-axis `"Frequency (count)"`, and use the title `"Small Business Owners: Age Distribution"`. 

# In[58]:


# Plot histogram of "AGE"
df_small_biz["AGE"].hist()
# Don't delete the code below 👇
plt.savefig("images/6-5-7.png", dpi=150)


# So, can we say the same thing about small business owners as we can about credit-fearful people?

# In[ ]:


with open("images/6-5-7.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.7", file)


# Let's take a look at the variance in the dataset.

# **Task 6.5.8:** Calculate the variance for all the features in `df_small_biz`, and create a Series `top_ten_var` with the 10 features with the largest variance.

# In[59]:


# Calculate variance, get 10 largest features
top_ten_var = df_small_biz.var().nlargest(n=10, keep='first')
top_ten_var


# In[ ]:



wqet_grader.grade("Project 6 Assessment", "Task 6.5.8", top_ten_var)


# We'll need to remove some outliers to avoid problems in our calculations, so let's trim them out.

# **Task 6.5.9:** Calculate the trimmed variance for the features in `df_small_biz`. Your calculations should not include the top and bottom 10% of observations. Then create a Series `top_ten_trim_var` with the 10 features with the largest variance.

# In[60]:


# Calculate trimmed variance
top_ten_trim_var = df_small_biz.apply(trimmed_var).sort_values().tail(10)
type(top_ten_trim_var)


# In[ ]:



wqet_grader.grade("Project 6 Assessment", "Task 6.5.9", top_ten_trim_var)


# Let's do a quick visualization of those values.

# **Task 6.5.10:** Use plotly express to create a horizontal bar chart of `top_ten_trim_var`. Be sure to label your x-axis `"Trimmed Variance [$]"`, the y-axis `"Feature"`, and use the title `"Small Business Owners: High Variance Features"`.

# In[61]:


# Create horizontal bar chart of `top_ten_trim_var`
fig = px.bar(x=top_ten_trim_var,y=top_ten_trim_var.index,title="Small Business Owners: High Variance Features")
fig.update_layout(xaxis_title="",yaxis_title="Feature")


# Don't delete the code below 👇
fig.write_image("images/6-5-10.png", scale=1, height=500, width=700)

fig.show()


# In[ ]:


with open("images/6-5-10.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.10", file)


# Based on this graph, which five features have the highest variance?

# **Task 6.5.11:** Generate a list `high_var_cols` with the column names of the  five features with the highest trimmed variance.

# In[62]:


high_var_cols =top_ten_trim_var.tail(5).index.to_list()
type(high_var_cols)


# In[ ]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.11", high_var_cols)


# ## Split

# Let's turn that list into a feature matrix.

# **Task 6.5.12:** Create the feature matrix `X`. It should contain the five columns in `high_var_cols`.

# In[63]:


X = df_small_biz[high_var_cols]
print("X shape:", X.shape)


# In[ ]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.12", list(X.shape))


# # Build Model

# Now that our data is in order, let's get to work on the model.

# ## Iterate

# **Task 6.5.13:** Use a `for` loop to build and train a K-Means model where `n_clusters` ranges from 2 to 12 (inclusive). Your model should include a `StandardScaler`. Each time a model is trained, calculate the inertia and add it to the list `inertia_errors`, then calculate the silhouette score and add it to the list `silhouette_scores`.

# <div class="alert alert-info" role="alert">
#     <b>Note:</b> For reproducibility, make sure you set the random state for your model to <code>42</code>. 
# </div>

# In[64]:


n_clusters = range(2,13)
inertia_errors = []
silhouette_scores = []

# Add `for` loop to train model and calculate inertia, silhouette score.
for n in n_clusters:
    model=make_pipeline(StandardScaler(),KMeans(n_clusters=n,random_state=42))
    model.fit(X)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    
    silhouette_scores.append(silhouette_score(X,model.named_steps["kmeans"].labels_))

print("Inertia:", inertia_errors[:11])
print()
print("Silhouette Scores:", silhouette_scores[:3])


# In[ ]:


wqet_grader.grade("Project 6 Assessment", "Task 6.5.13", list(inertia_errors))


# Just like we did in the previous module, we can start to figure out how many clusters we'll need with a line plot based on Inertia.

# **Task 6.5.14:** Use plotly express to create a line plot that shows the values of `inertia_errors` as a function of `n_clusters`. Be sure to label your x-axis `"Number of Clusters"`, your y-axis `"Inertia"`, and use the title `"K-Means Model: Inertia vs Number of Clusters"`.

# In[65]:


# Create line plot of `inertia_errors` vs `n_clusters`
fig = px.line(x=n_clusters,y=inertia_errors,title="K-Means Model: Inertia vs Number of Clusters",)
fig.update_layout(xaxis_title="Number of Clusters",yaxis_title="Inertia")
# Don't delete the code below 👇
fig.write_image("images/6-5-14.png", scale=1, height=500, width=700)

fig.show()


# In[ ]:


with open("images/6-5-14.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.14", file)


# And let's do the same thing with our Silhouette Scores.

# **Task 6.5.15:** Use plotly express to create a line plot that shows the values of `silhouette_scores` as a function of `n_clusters`. Be sure to label your x-axis `"Number of Clusters"`, your y-axis `"Silhouette Score"`, and use the title `"K-Means Model: Silhouette Score vs Number of Clusters"`.

# In[66]:


# Create a line plot of `silhouette_scores` vs `n_clusters`
fig = px.line(x=n_clusters,
              y=silhouette_scores,
              title="K-Means Model: Silhouette Score vs Number of Clusters")
fig.update_layout(xaxis_title="Number of Clusters",yaxis_title="Silhouette Score")
# Don't delete the code below 👇
fig.write_image("images/6-5-15.png", scale=1, height=500, width=700)

fig.show()


# In[ ]:


with open("images/6-5-15.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.15", file)


# How many clusters should we use? When you've made a decision about that, it's time to build the final model.

# **Task 6.5.16:** Build and train a new k-means model named `final_model`. The number of clusters should be `3`.

# <div class="alert alert-info" role="alert">
#     <b>Note:</b> For reproducibility, make sure you set the random state for your model to <code>42</code>. 
# </div>

# In[67]:


final_model =make_pipeline(StandardScaler(),KMeans(n_clusters=3,random_state=42))
final_model.fit(X)


# In[ ]:


# match_steps, match_hyperparameters, prune_hyperparameters should all be True

wqet_grader.grade("Project 6 Assessment", "Task 6.5.16", final_model)


# # Communicate

# Excellent! Let's share our work! 

# **Task 6.5.17:** Create a DataFrame `xgb` that contains the mean values of the features in `X` for the 3 clusters in your `final_model`.

# In[68]:


labels = final_model.named_steps["kmeans"].labels_
xgb = X.groupby(labels).mean()
xgb


# In[ ]:



wqet_grader.grade("Project 6 Assessment", "Task 6.5.17", xgb)


# As usual, let's make a visualization with the DataFrame.

# **Task 6.5.18:** Use plotly express to create a side-by-side bar chart from `xgb` that shows the mean of the features in `X` for each of the clusters in your `final_model`. Be sure to label the x-axis `"Cluster"`, the y-axis `"Value [$]"`, and use the title `"Small Business Owner Finances by Cluster"`.

# In[69]:


# Create side-by-side bar chart of `xgb`
fig = px.bar(xgb,barmode="group",title="Mean Household Finances by Cluster")
fig.update_layout(xaxis_title="Cluster",yaxis_title="Value [$]")

# Don't delete the code below 👇
fig.write_image("images/6-5-18.png", scale=1, height=500, width=700)

fig.show()


# In[ ]:


with open("images/6-5-18.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.18", file)


# Remember what we did with higher-dimension data last time? Let's do the same thing here.

# **Task 6.5.19:** Create a `PCA` transformer, use it to reduce the dimensionality of the data in `X` to 2, and then put the transformed data into a DataFrame named `X_pca`. The columns of `X_pca` should be named `"PC1"` and `"PC2"`.

# In[72]:


pca = PCA(n_components=2,random_state=42)

# Transform `X`
X_t = pca.fit_transform(X)

# Put `X_t` into DataFrame
X_pca = pd.DataFrame(X_t,columns=["PC1","PC2"])

print("X_pca shape:", X_pca.shape)
X_pca.head()


# In[73]:



wqet_grader.grade("Project 6 Assessment", "Task 6.5.19", X_pca)


# Finally, let's make a visualization of our final DataFrame.

# **Task 6.5.20:** Use plotly express to create a scatter plot of `X_pca` using seaborn. Be sure to color the data points using the labels generated by your `final_model`. Label the x-axis `"PC1"`, the y-axis `"PC2"`, and use the title `"PCA Representation of Clusters"`.

# In[ ]:


# Create scatter plot of `PC2` vs `PC1`

# Don't delete the code below 👇
fig.write_image("images/6-5-20.png", scale=1, height=500, width=700)

fig.show()


# In[ ]:


with open("images/6-5-20.png", "rb") as file:
    wqet_grader.grade("Project 6 Assessment", "Task 6.5.20", file)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
