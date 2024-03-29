# -*- coding: utf-8 -*-
"""McDonalds Case Study.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wLo0p-KPEvYQB2hP6UJGrHYC8Ey4PuVM
"""

import pandas as pd

df = pd.read_csv('mcdonalds.csv')

print("Variable Names:")
print(df.columns)

print("\nSample Size:")
print(len(df))

print("\nFirst Three Rows:")
print(df.head(3))

segmentation_data = df.iloc[:, :11]

numeric_segmentation_data = segmentation_data.apply(lambda x: x.eq('Yes').astype(int))

print("Average Value of Transformed Segmentation Variables:")
print(numeric_segmentation_data.mean())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_segmentation_data)

pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 8))
plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.5)
plt.title('Perceptual Map - Principal Components Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i, txt in enumerate(df.index):
    plt.annotate(txt, (pc_df['PC1'][i], pc_df['PC2'][i]))

plt.show()

explained_variance_ratio = pca.explained_variance_ratio_
print(f'Explained Variance Ratio - PC1: {explained_variance_ratio[0]:.2f}')
print(f'Explained Variance Ratio - PC2: {explained_variance_ratio[1]:.2f}')

factor_loadings = pca.components_.T

print("Factor Loadings:")
print(factor_loadings)

rotated_vars = np.dot(numeric_segmentation_data.values, factor_loadings)

plt.figure(figsize=(8, 8))
plt.scatter(rotated_vars[:, 0], rotated_vars[:, 1], color='grey', label='Consumers')

for i in range(len(factor_loadings)):
    plt.arrow(0, 0, factor_loadings[i, 0], factor_loadings[i, 1], color='red', alpha=0.5, width=0.02, head_width=0.1)
    plt.text(factor_loadings[i, 0], factor_loadings[i, 1], segmentation_data.columns[i], color='red', fontsize=8)

plt.xlabel('Rotated Principal Component 1')
plt.ylabel('Rotated Principal Component 2')
plt.title('Rotated and Projected Consumers with Segmentation Variables Arrows')

plt.show()

#Using k-Means

from sklearn.cluster import KMeans

num_segments = 4

kmeans = KMeans(n_clusters=num_segments, random_state=42)
segment_labels = kmeans.fit_predict(rotated_vars)

pc_df['Segment'] = segment_labels

print(pc_df.head())

!pip install scikit-learn

from sklearn.metrics import silhouette_score

inertia_values = []

min_clusters = 2
max_clusters = 8

for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(numeric_segmentation_data)

    # Calculate sum of squared distances (inertia) for the current clustering
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(min_clusters, max_clusters + 1), inertia_values, marker='o')
plt.title('Scree Plot - Sum of Squared Distances')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

from sklearn.utils import resample

n_bootstraps = 200

n_init = 10

min_clusters = 2
max_clusters = 8

cluster_labels_bootstrap = np.zeros((n_bootstraps, max_clusters - min_clusters + 1, len(numeric_segmentation_data)))

for i in range(n_bootstraps):
    # Generate a bootstrap sample
    bootstrap_sample = resample(numeric_segmentation_data, replace=True, random_state=i)

    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=i)
        kmeans.fit(bootstrap_sample)

        # Store cluster labels
        cluster_labels_bootstrap[i, num_clusters - min_clusters, :] = kmeans.labels_

stability_scores = np.zeros((max_clusters - min_clusters + 1,))
for num_clusters in range(min_clusters, max_clusters + 1):
    # Calculate the proportion of times each pair of observations is clustered together across bootstrap samples
    pair_agreement = np.sum(cluster_labels_bootstrap[:, num_clusters - min_clusters, :] == cluster_labels_bootstrap[:, num_clusters - min_clusters, None, :], axis=0) / n_bootstraps

    # Calculate the stability score for the current number of clusters
    stability_scores[num_clusters - min_clusters] = np.mean(pair_agreement)

plt.figure(figsize=(8, 5))
plt.plot(range(min_clusters, max_clusters + 1), stability_scores, marker='o')
plt.title('Global Stability Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Stability Score')
plt.show()

stability_scores = cluster_labels_bootstrap[:, :, 0]

plt.figure(figsize=(8, 5))
plt.boxplot(stability_scores, labels=range(min_clusters, max_clusters + 1))
plt.title('Global Stability Boxplot for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Stability Score')
plt.show()

pip install seaborn

import seaborn as sns

sns.set(style="white")

sns.pairplot(pc_df, hue='Segment', palette='Dark2', markers=["o", "s", "D", "P"], diag_kind="kde")

plt.show()

from sklearn.metrics import pairwise_distances

pairwise_distances_within_segments = []
for segment in range(num_segments):
    segment_data = rotated_vars[segment_labels == segment]
    pairwise_distances_within_segments.append(pairwise_distances(segment_data))

similarity_values = []
segment_pairs = []

for i in range(num_segments):
    for j in range(i + 1, num_segments):
        min_data_points = min(pairwise_distances_within_segments[i].shape[0], pairwise_distances_within_segments[j].shape[0])
        similarity = 1 - pairwise_distances_within_segments[i][:min_data_points, :min_data_points] / pairwise_distances_within_segments[j][:min_data_points, :min_data_points]

        # Flatten the similarity matrix and store values and corresponding segment pairs
        similarity_values.extend(similarity.flatten())
        segment_pairs.extend([f'Segment {i+1} vs Segment {j+1}' for _ in range(min_data_points**2)])

data = pd.DataFrame({'Segment Pairs': segment_pairs, 'Similarity Values': similarity_values})

plt.figure(figsize=(10, 8))
sns.boxplot(x='Segment Pairs', y='Similarity Values', data=data)

plt.title('Gorge Plot - Similarity Values Between Segments')
plt.xlabel('Segment Pairs')
plt.ylabel('Similarity Value')

plt.show()

pip install scikit-learn-extra

pip install segmentation-models

from sklearn.metrics import adjusted_rand_score

num_segments_range = range(min_clusters, max_clusters + 1)

adj_rand_indices = np.zeros((len(num_segments_range), len(num_segments_range)))
for i, num_segments_i in enumerate(num_segments_range):
    for j, num_segments_j in enumerate(num_segments_range):
        if i <= j:
            labels_i = cluster_labels_bootstrap[i, num_segments_i - min_clusters, :]
            labels_j = cluster_labels_bootstrap[j, num_segments_j - min_clusters, :]
            adj_rand_indices[i, j] = adjusted_rand_score(labels_i, labels_j)
            adj_rand_indices[j, i] = adj_rand_indices[i, j]

plt.figure(figsize=(12, 8))
plt.imshow(adj_rand_indices, cmap='Blues', aspect='auto', extent=[min_clusters-0.5, max_clusters+0.5, min_clusters-0.5, max_clusters+0.5])

plt.title('Adjusted Rand Index Across Solutions')
plt.xlabel('Number of Segments')
plt.ylabel('Number of Segments')

plt.colorbar(label='Adjusted Rand Index')
plt.show()

from scipy.stats import mode

segment_stabilities = np.zeros((num_segments, len(cluster_labels_bootstrap)))

for i in range(num_segments):
    for j in range(len(cluster_labels_bootstrap)):
        labels_i = cluster_labels_bootstrap[j, i, :]
        segment_stabilities[i, j] = np.mean(labels_i == mode(labels_i)[0])

plt.figure(figsize=(10, 6))
for i in range(num_segments):
    plt.plot(segment_stabilities[i, :], label=f'Segment {i+1}')

plt.title('Segment Level Stability Within Solutions (SLSW)')
plt.xlabel('Replication')
plt.ylabel('Segment Stability')
plt.ylim(0, 1)
plt.xticks(range(len(cluster_labels_bootstrap)), range(1, len(cluster_labels_bootstrap)+1))
plt.legend()
plt.show()

# Using Mixtures of Distributions

pip install mixmod

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

min_segments = 2
max_segments = 8
n_init = 10
n_bootstraps = 200

inertia_values = []
silhouette_scores = []
cluster_labels_bootstrap = np.zeros((n_bootstraps, max_segments - min_segments + 1, len(numeric_segmentation_data)))

for num_segments in range(min_segments, max_segments + 1):
    model = GaussianMixture(n_components=num_segments, n_init=n_init)
    model.fit(numeric_segmentation_data)

    inertia_values.append(model.bic(numeric_segmentation_data))

    silhouette_scores.append(silhouette_score(numeric_segmentation_data, model.predict(numeric_segmentation_data)))

    # Bootstrap stability analysis
    for i in range(n_bootstraps):

        bootstrap_sample = resample(numeric_segmentation_data, replace=True, random_state=i)

        bootstrap_model = GaussianMixture(n_components=num_segments, n_init=n_init)
        bootstrap_model.fit(bootstrap_sample)

        cluster_labels_bootstrap[i, num_segments - min_segments, :] = bootstrap_model.predict(bootstrap_sample)

plt.figure(figsize=(8, 5))
plt.plot(range(min_segments, max_segments + 1), inertia_values, marker='o')
plt.title('Scree Plot - BIC (Bayesian Information Criterion)')
plt.xlabel('Number of Segments')
plt.ylabel('BIC')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(min_segments, max_segments + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score Plot')
plt.xlabel('Number of Segments')
plt.ylabel('Silhouette Score')
plt.show()

stability_scores = np.zeros((max_segments - min_segments + 1,))
for num_segments in range(min_segments, max_segments + 1):
    pair_agreement = np.sum(cluster_labels_bootstrap[:, num_segments - min_segments, :] == cluster_labels_bootstrap[:, num_segments - min_segments, None, :], axis=0) / n_bootstraps
    stability_scores[num_segments - min_segments] = np.mean(pair_agreement)

plt.figure(figsize=(8, 5))
plt.plot(range(min_segments, max_segments + 1), stability_scores, marker='o')
plt.title('Global Stability Scores for Different Numbers of Segments')
plt.xlabel('Number of Segments')
plt.ylabel('Stability Score')
plt.show()

from sklearn.mixture import GaussianMixture

min_components = 2
max_components = 10  # Adjust as needed

aic_values = []
bic_values = []
icl_values = []

for num_components in range(min_components, max_components + 1):
    model = GaussianMixture(n_components=num_components, random_state=42)
    model.fit(numeric_segmentation_data)

    aic_values.append(model.aic(numeric_segmentation_data))
    bic_values.append(model.bic(numeric_segmentation_data))

plt.figure(figsize=(10, 6))
plt.plot(range(min_components, max_components + 1), aic_values, label='AIC', marker='o')
plt.plot(range(min_components, max_components + 1), bic_values, label='BIC', marker='o')

plt.title('Information Criteria for Different Numbers of Components')
plt.xlabel('Number of Components')
plt.ylabel('Information Criteria Value')
plt.legend()
plt.show()

from sklearn.mixture import GaussianMixture

num_clusters_kmeans = 4  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters_kmeans, random_state=42)
cluster_labels_kmeans = kmeans.fit_predict(numeric_segmentation_data)

num_components_gmm = 4  # Adjust as needed
gmm = GaussianMixture(n_components=num_components_gmm, random_state=42)
gmm.fit(numeric_segmentation_data)
component_memberships_gmm = gmm.predict_proba(numeric_segmentation_data)

comparison_df = pd.DataFrame(component_memberships_gmm, columns=[f'Component {i+1}' for i in range(num_components_gmm)])
comparison_df['Cluster Membership (k-means)'] = cluster_labels_kmeans

crosstab_df = pd.crosstab(comparison_df['Cluster Membership (k-means)'], columns=comparison_df.iloc[:, :-1].idxmax(axis=1))

print(crosstab_df)

plt.figure(figsize=(10, 6))
plt.imshow(crosstab_df.values, cmap='Blues', aspect='auto', interpolation='none')
plt.colorbar(label='Count of Members')
plt.xlabel('Component (GMM)')
plt.ylabel('Cluster (k-means)')
plt.title('Comparison of GMM Components and k-means Clusters')
plt.xticks(np.arange(num_components_gmm), labels=[f'Component {i+1}' for i in range(num_components_gmm)])
plt.show()

#Using Mixtures of Regression Models

from sklearn.linear_model import LinearRegression

independent_variables = df[['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']]

print(df.head())

original_like_counts = df['Like'].value_counts(sort=False)
print(original_like_counts)

df['Like.n'] = 6 - df['Like'].astype(int)

new_like_counts = df['Like.n'].value_counts(sort=False)
print(new_like_counts)

import statsmodels.api as sm

categorical_columns = ['Gender']

numeric_columns = ['Age', 'VisitFrequency']

print(df.dtypes)

df['VisitFrequency'] = pd.to_numeric(df['VisitFrequency'], errors='coerce')

print(df.dtypes)

from sklearn.impute import SimpleImputer

num_components = 2

imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
independent_variables_imputed = imputer.fit_transform(independent_variables)

gmm = GaussianMixture(n_components=num_components, random_state=42)
gmm.fit(independent_variables_imputed)

pip install pymer4

import pymer4

pip install seaborn matplotlib

import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

num_components = 2
gmm = GaussianMixture(n_components=num_components, random_state=42)
gmm.fit(independent_variables_imputed)

cluster_labels = gmm.predict(independent_variables_imputed)

dependent_variable = df['Like']

# Fit linear regression models for each cluster
regression_models = {}
for cluster in range(num_components):
    cluster_data = independent_variables_imputed[cluster_labels == cluster]
    dependent_variable_cluster = dependent_variable[cluster_labels == cluster]

    model = LinearRegression().fit(cluster_data, dependent_variable_cluster)
    regression_models[cluster] = model

    print(f"\nRegression Coefficients for Cluster {cluster + 1}:")
    for feature, coef in zip(independent_variables.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

attributes_data = segmentation_data.iloc[:, 1:]

attributes_data_encoded = pd.get_dummies(attributes_data, columns=attributes_data.select_dtypes(include=['object']).columns)

linkage_matrix = hierarchy.linkage(attributes_data_encoded.T, method='ward')

plt.figure(figsize=(12, 8))
dendrogram = hierarchy.dendrogram(linkage_matrix, orientation='top', leaf_rotation=90)

num_clusters = 4
cluster_labels = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward').fit_predict(attributes_data_encoded.T)

if len(cluster_labels) == len(attributes_data_encoded.columns):

    attributes_data_encoded.columns = [f'{col}_Cluster{cluster}' for col, cluster in zip(attributes_data_encoded.columns, cluster_labels)]

    numeric_cluster_labels = [int(label.split('Cluster')[-1]) for label in attributes_data_encoded.columns]

    plt.figure(figsize=(12, 8))
    sns.heatmap(attributes_data_encoded.groupby(numeric_cluster_labels, axis=1).mean().T, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Segment Profile Plot')
    plt.show()
else:
    print("Error: Length of cluster labels does not match the number of attributes.")

from scipy.cluster.hierarchy import linkage, dendrogram

MD_x_transposed = df.T

MD_x_encoded = pd.get_dummies(MD_x_transposed)

linkage_matrix = linkage(MD_x_encoded, method='average')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, orientation='top', labels=MD_x_transposed.index, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Segment', data=pc_df, palette='Dark2', s=100, alpha=0.8)
plt.title('Segment Separation Plot - Principal Components Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i in range(num_segments):
    segment_center = pc_df[pc_df['Segment'] == i][['PC1', 'PC2']].mean()
    plt.scatter(segment_center[0], segment_center[1], marker='o', color='black', s=200, label=f'Segment {i + 1}')

plt.legend()
plt.show()

color_segments = [1, 2]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Segment', data=pc_df, palette='Dark2', s=100, alpha=0.8)

for i in range(num_segments):
    segment_center = pc_df[pc_df['Segment'] == i][['PC1', 'PC2']].mean()
    if i in color_segments:
        plt.scatter(segment_center[0], segment_center[1], marker='o', s=200, label=f'Segment {i + 1}', color=f'C{i}')
    else:
        plt.scatter(segment_center[0], segment_center[1], marker='o', color='black', s=200, label=f'Segment {i + 1}')

plt.legend()
plt.title('Segment Separation Plot - Two Different Colors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

from statsmodels.graphics.mosaicplot import mosaic

merged_df = pd.merge(pc_df[['Segment']], df[['Like']], left_index=True, right_index=True)

plt.figure(figsize=(10, 8))
mosaic(merged_df, ['Segment', 'Like'], title='Segment Membership and I LIKE IT', gap=0.02, properties=lambda key: {'color': 'C0' if key[1] == 'No' else 'C1'})
plt.xlabel('Segment Number')
plt.show()

merged_df_gender = pd.merge(pc_df[['Segment']], df[['Gender']], left_index=True, right_index=True)

plt.figure(figsize=(10, 8))
mosaic(merged_df_gender, ['Segment', 'Gender'], title='Gender Distribution Across Segments', gap=0.02)
plt.xlabel('Segment Number')
plt.show()

merged_df_age = pd.merge(pc_df[['Segment']], df[['Age']], left_index=True, right_index=True)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Segment', y='Age', data=merged_df_age, width=0.5, notch=True)
plt.title('Parallel Box-and-Whisker Plot of Age by Segment')
plt.xlabel('Segment Number')
plt.ylabel('Age')
plt.show()

