import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

z_score = 1.96
population_proportion = 0.5
margin_error = 0.05
strata_param = 5
cluster_param = 10

data = pd.read_csv('Creditcard_data.csv')
X_data = data.drop(['Class'], axis=1)
y_data = data['Class']

smote_model = SMOTE()
X_resampled, y_resampled = smote_model.fit_resample(X_data, y_data)

sample_simple_size = int((z_score**2 * population_proportion * (1 - population_proportion)) / margin_error**2)
sample_stratified_size = int((z_score**2 * population_proportion * (1 - population_proportion)) / (margin_error / strata_param)**2)
sample_cluster_size = int((z_score**2 * population_proportion * (1 - population_proportion)) / (margin_error / cluster_param)**2)
sample_systematic_size = len(X_resampled) // 20

ml_models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(),
    'KNeighbors': KNeighborsClassifier()
}

simple_sample = X_resampled.sample(n=sample_simple_size, random_state=42)

strata_labels = pd.qcut(y_resampled, q=strata_param, labels=False, duplicates='drop')
stratified_sample = X_resampled.groupby(strata_labels, group_keys=False).apply(
    lambda x: x.sample(min(len(x), sample_stratified_size // strata_param), random_state=42)
)

kmeans_model = KMeans(n_clusters=cluster_param, random_state=42)
cluster_labels = kmeans_model.fit_predict(X_resampled)
X_resampled['cluster'] = cluster_labels
cluster_sample = X_resampled.groupby('cluster', group_keys=False).apply(
    lambda x: x.sample(min(len(x), sample_cluster_size // cluster_param), random_state=42)
)

systematic_interval = len(X_resampled) // sample_systematic_size
systematic_indices = np.arange(0, len(X_resampled), step=systematic_interval)
systematic_sample = X_resampled.iloc[systematic_indices]

multistage_sample = stratified_sample.groupby(strata_labels, group_keys=False).apply(
    lambda x: x.sample(min(len(x), sample_stratified_size // strata_param // 2), random_state=42)
)

results = []
sampling_methods = ['Simple Random', 'Stratified', 'Cluster', 'Systematic', 'Multistage']
samples = [simple_sample, stratified_sample, cluster_sample, systematic_sample, multistage_sample]

for method, sample in zip(sampling_methods, samples):
    X_sampled = sample.drop(['cluster'], axis=1, errors='ignore')
    y_sampled = y_resampled.loc[X_sampled.index]
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
    
    for model_name, model in ml_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results.append({'Model': model_name, 'Sampling Technique': method, 'Accuracy': acc})

results_df = pd.DataFrame(results)
pivot_df = results_df.pivot(index='Model', columns='Sampling Technique', values='Accuracy')

pivot_df.to_csv('model_sampling_results.csv')
print(pivot_df)
