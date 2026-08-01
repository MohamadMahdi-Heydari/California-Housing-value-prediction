# California-Housing-value-prediction
A beginner-friendly machine learning project implementing Linear Regression on the California Housing dataset.

# California Housing Price Prediction using Linear Regression

## Overview

This project implements a machine learning regression model to predict California housing prices using the California Housing dataset.

The main goal of this project is to build a complete machine learning workflow, including data preprocessing, feature selection, model training, evaluation, and visualization.

A Linear Regression model is used to learn the relationship between different housing features and the median house value. The project also includes exploratory data analysis using correlation analysis and data visualization techniques.

---

## Dataset

The dataset used in this project is the California Housing dataset.

Each row represents a district in California, and the model tries to predict the median house value of that district.

The dataset contains information such as:

- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Median house value (Target)

The `ocean_proximity` feature was removed because it is a categorical feature and this project focuses on numerical data and Linear Regression.

---

## Project Workflow

The project follows these steps:

1. Import required libraries
2. Load the dataset using Pandas
3. Handle missing values
4. Select numerical features
5. Split the data into training and testing sets
6. Train a Linear Regression model
7. Make predictions
8. Evaluate the model using R² Score
9. Visualize predictions and feature correlations

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Data Preprocessing

First, the dataset is loaded:

```python
df = pd.read_csv("housing.csv")
```

The dataset contains some missing values, so they are removed:

```python
df = df.dropna()
```

The target variable is:

```python
y = df["median_house_value"]
```

The input features are:

```python
x = df.drop(columns=["median_house_value", "ocean_proximity"])
```

The categorical column `ocean_proximity` was removed because Linear Regression requires numerical input.

---

## Model Training

The dataset is divided into training and testing data:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

80% of the data is used for training and 20% is used for testing.

The model is trained using Scikit-learn Linear Regression:

```python
model = linear_model.LinearRegression()

model.fit(x_train, y_train)
```

---

## Prediction

After training, the model predicts house prices:

```python
y_pred = model.predict(x_test)
```

The model can also predict new custom inputs:

```python
a = np.array([
[-122.01, 37.39, 26, 2500, 962, 2374, 879, 3.5586]
])

prediction = model.predict(a)
```

---

## Model Evaluation

The performance of the model is measured using the R² Score.

```python
r2_score(y_test, y_pred)
```

The obtained result:

```
R² Score ≈ 0.64
```

This means the Linear Regression model explains around 64% of the variation in house prices.

A score around this range is expected because house prices depend on many complex factors that are not included in the dataset.

---

## Visualization

### Predicted vs Actual Values

This plot compares the real house prices with the values predicted by the model.

The red line represents the ideal prediction line where:

```
Predicted Value = Actual Value
```

The closer the points are to this line, the better the model predictions are.

![Predicted vs Actual](https://github.com/MohamadMahdi-Heydari/California-Housing-value-prediction/blob/main/Figure_1.png](https://github.com/MohamadMahdi-Heydari/California-Housing-value-prediction/blob/main/prediction.png)


---

### Correlation Heatmap

The correlation heatmap shows the relationship between numerical features.

Values closer to:

- `1` indicate a strong positive relationship
- `-1` indicate a strong negative relationship
- `0` indicate little or no relationship

![Correlation Heatmap](images/correlation_heatmap.png](https://github.com/MohamadMahdi-Heydari/California-Housing-value-prediction/blob/main/Figure_1.png)

---

## Example Prediction

The model can estimate a house value using input features:

Input:

```
Longitude: -122.01
Latitude: 37.39
Housing Age: 26
Total Rooms: 2500
Total Bedrooms: 962
Population: 2374
Households: 879
Median Income: 3.5586
```

The trained model returns an estimated house price.

---

## Limitations

Although Linear Regression is simple and useful for understanding machine learning concepts, it has some limitations:

- It assumes a linear relationship between features and target.
- Housing prices are affected by many external factors.
- The model cannot fully capture complex patterns.

More advanced models such as Random Forest, Gradient Boosting, or XGBoost could achieve better performance.

---

## Future Improvements

Possible improvements:

- Apply feature scaling
- Use One-Hot Encoding for categorical features
- Try more advanced regression algorithms
- Perform hyperparameter tuning
- Add more evaluation metrics such as MAE and RMSE

---

## Conclusion

This project demonstrates the complete process of building a regression model from raw data to prediction.

Through this project, the main concepts of machine learning regression were practiced:

- Data preprocessing
- Feature selection
- Model training
- Model evaluation
- Data visualization

The project provides a foundation for developing more advanced machine learning models in the future.
