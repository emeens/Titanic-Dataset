#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_excel('CS379T-Week-1-IP.xls')
print(data.isnull().sum())
data.drop(['name','ticket','cabin','boat', 'body', 'home.dest'], axis=1, inplace=True) #dropping irrelevent columns

#Convert categorical variables to numerical using label encoding
le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])
data["embarked"] = le.fit_transform(data["embarked"].astype(str))

#Fill in missing values in the "age" and "embarked" features using mean and most frequent imputation
#dropping values, converting to numerical, and filling in missing values is all part of cleaning the data.
data["age"].fillna(data["age"].mean(), inplace=True)
data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)
data["fare"].fillna(data["fare"].mode()[0], inplace=True)


plt.figure(figsize=(25, 7))
ax = plt.subplot()
ax.scatter(data[data['survived'] == 1]['age'], data[data['survived'] == 1]['fare'], c='green', s=data[data['survived'] == 1]['fare'])
ax.scatter(data[data['survived'] == 0]['age'], data[data['survived'] == 0]['fare'], c='red', s=data[data['survived'] == 0]['fare'])

y = data["survived"]
x = data[["pclass", "sex", "age", "fare"]]


#Training the K-means model
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

plt.scatter(x['age'], x['fare'], c=kmeans.labels_, cmap='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='k')
plt.xlabel('age')
plt.ylabel('fare')
plt.title('Titanic Clustering (K-means)')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Training the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(y_pred)
