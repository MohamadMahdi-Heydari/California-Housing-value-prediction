import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")
df = df.dropna()
model = linear_model.LinearRegression()

x = df.drop(columns=['median_house_value',"ocean_proximity"])
y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

a = np.array([[-122.01, 37.39, 26, 2500, 962, 2374, 879, 3.5586]])
              
a_pred = model.predict(a)

print(a_pred)
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.grid(alpha=0.3)
plt.scatter(y_test,y_pred,alpha=0.4, s=40)
plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()], color='red',lw=6
)
plt.xlabel("Actual (y_test)")
plt.ylabel("Predicted (y_pred)")
plt.title("Predicted vs Actual")

corr = df.drop(columns=["ocean_proximity"]).corr()
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)

plt.show()
