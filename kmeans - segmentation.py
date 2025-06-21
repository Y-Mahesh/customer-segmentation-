import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = pd.DataFrame({
    'Age': [25, 34, 45, 31, 40, 23, 36],
    'Income': [40000, 60000, 80000, 50000, 70000, 30000, 65000],
    'Spending Score': [60, 65, 45, 70, 48, 80, 55]
})

# Select features
X = data[['Age', 'Income', 'Spending Score']]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize
sns.scatterplot(data=data, x='Income', y='Spending Score', hue='Cluster')
plt.title("Customer Segments")
plt.show()
