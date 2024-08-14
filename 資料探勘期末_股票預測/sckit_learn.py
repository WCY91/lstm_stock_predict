import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load and clean data
data = pd.read_csv('data\\2330.csv')
data.dropna(inplace=True)

# Improved correlation matrix
correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('correlation_matrix.png')
plt.show()

# Feature selection using RandomForestRegressor
x = data[['Open', 'High', 'Low', 'Close', 'Adj_Close']]
y = data['Volume']

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(x_scaled, y)

feature_importance = pd.Series(rf_model.feature_importances_, index=x.columns)
feature_importance.plot(kind='bar', color='skyblue')
selected_features_random = x.columns[feature_importance != 0]
plt.title('Feature Importance', fontsize=16)
plt.savefig('feature_importance_rf.png')
plt.show()

# Using Lasso for feature selection
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(x_scaled, y)

lasso_coef = pd.Series(lasso.coef_, index=x.columns)
lasso_coef.plot(kind='bar', color='skyblue')
plt.title('Lasso Coefficients', fontsize=16)
plt.savefig('lasso_coefficients.png')
plt.show()

selected_features_lasso = x.columns[lasso.coef_ != 0]
print("Selected Features using Lasso:", selected_features_lasso)

# Using Elastic Net for feature selection
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
elastic_net.fit(x_scaled, y)

elastic_net_coef = pd.Series(elastic_net.coef_, index=x.columns)
elastic_net_coef.plot(kind='bar', color='skyblue')
plt.title('Elastic Net Coefficients', fontsize=16)
plt.savefig('elastic_net_coefficients.png')
plt.show()

# Using Recursive Feature Elimination (RFE) for feature selection
lin_reg_model = LinearRegression()
rfe = RFE(lin_reg_model, n_features_to_select=3)
rfe.fit(x_scaled, y)

selected_features_rfe = x.columns[rfe.support_]
print("Selected Features using RFE:", selected_features_rfe)

# Using Gradient Boosting for feature importance
gb_model = GradientBoostingRegressor()
gb_model.fit(x_scaled, y)

gb_importance = pd.Series(gb_model.feature_importances_, index=x.columns)
gb_importance.plot(kind='bar', color='skyblue')
plt.title('Gradient Boosting Feature Importance', fontsize=16)
plt.savefig('gb_feature_importance.png')
plt.show()

# Summary of selected features
print("Selected Features using RandomForestRegressor:", x.columns[rf_model.feature_importances_ > 0.1])
print("Selected Features using Elastic Net:", x.columns[elastic_net.coef_ != 0])

# Train-test split with selected features from Lasso
x_train, x_test, y_train, y_test = train_test_split(x[selected_features_random], y, test_size=0.3, random_state=42)

# Standardize the selected features
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Linear Regression
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(x_test_scaled)
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Cross-validation score
cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5)
print("Cross-Validation R^2 Scores:", cv_scores)
print("Mean Cross-Validation R^2 Score:", np.mean(cv_scores))

# Plot Actual vs Predicted Volume
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Volume', color='blue')
plt.plot(y_pred, label='Predicted Volume', color='red')
plt.title('Actual vs Predicted Volume', fontsize=16)
plt.xlabel('Index', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.legend()
plt.savefig('actual_vs_predicted_volume.png')
plt.show()

# Prediction changes analysis
threshold = 0.05  
predictions = model.predict(x_test_scaled)
actual_changes = y_test.values
predicted_changes = predictions

for actual, predicted in zip(actual_changes, predicted_changes):
    change = (predicted - actual) / actual
    if change > threshold:
        print("Stock will rise.")
    elif change < -threshold:
        print("Stock will fall.")
    else:
        print("Stock will remain relatively stable.")
