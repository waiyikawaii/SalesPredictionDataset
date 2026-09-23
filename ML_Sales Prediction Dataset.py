import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#load the dataset
df=pd.read_excel(r"D:\Finlatics\Machine Learning\MLResearch\MLResearch\Sales Prediction Dataset\advertising_sales_data.xlsx", sheet_name="original")
print(df.head())
print(df.info())
print(df.isnull().sum())

#handle the 2 missing Radio values by dropping those rows
df=df.dropna().reset_index(drop=True)
print(df.shape)

'Q1: What is the average amount spent on TV advertising in the dataset?'
print(f"Average TV spend: ${np.mean(df['TV']):.2f}")

'Q2: What is the correlation between radio advertising expenditure and product sales?'
corr=df["Radio"].corr(df["Sales"])
print(f"Correlation (Radio, Sales): {corr:.4f}")

'Q3: Which advertising medium has the highest impact on sales based on the dataset?'
#correlation of each predictor with Sales
print("Correlation with Sales:")
print(df.corr(numeric_only=True)["Sales"].sort_values(ascending=False))
print()

#multiple linear regression
x=df[["TV", "Radio", "Newspaper"]]
y=df["Sales"]

model= LinearRegression().fit(x, y)
coefs=pd.Series(model.coef_, index=x.columns)
print(coefs)
print("Intercept:", model.intercept_)

#plot a heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="PuBuGn",fmt=".2f",vmin=-1, vmax=1,linewidths=0.5,square=True)
plt.title("Correlation Heatmap — Advertising vs Sales")
plt.tight_layout()
plt.show()

'Q4: Plot a linear regression line that includes all variables (TV, Radio, Newspaper) to predict Sales, and visualize the model\'s predictions against the actual sales values.'
#split into training and testing sets (80/20)
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

#build and train the multiple linear regression model
model=LinearRegression()
model.fit(X_train, y_train)

#predict sales on the test set
y_pred=model.predict(X_test)


#evaluate the model
r2=r2_score(y_test, y_pred)
b=model.intercept_
coefs=pd.Series(model.coef_, index=x.columns)
print(f"R² (test): {r2:.4f}")
print(f"Intercept: {b:.4f}")
print(coefs)

#build formula text
formula=f"Sales = {b:.3f}"
for name, value in coefs.items():
    formula += f" + {value:.4f}({name})"

#build coefficient text
coef_text=""
for name, value in coefs.items():
    coef_text += f"{name}: {value:.4f}\n"
    
#visualize the actual vs predict graph
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()],"r--", linewidth=2, label="Perfect prediction")
plt.text(2, 19,f"R² = {r2:.3f}\n{formula}\n\nCoefficients:\n{coef_text}",fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales — Multiple Linear Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

'Q5: How would sales be predicted for a new set of advertising expenditures: $200 on TV, $40 on Radio, and $50 on Newspaper?'
final_model = LinearRegression().fit(x, y)

new_data = pd.DataFrame({"TV": [200], "Radio": [40], "Newspaper": [50]})
prediction = final_model.predict(new_data)
print(f"Predicted Sales: {prediction[0]:.2f}")

'Q6: How does the performance of the linear regression model change when the dataset is normalized?'
#UNNORMALIZED MODEL
model_raw=LinearRegression().fit(X_train, y_train)
y_pred_raw=model_raw.predict(X_test)
r2_raw=r2_score(y_test, y_pred_raw)

print("------------")
print("UNNORMALIZED")
print("------------")
print(f"R²: {r2_raw:.6f}")
print(f"Intercept: {model_raw.intercept_:.4f}")
print(pd.Series(model_raw.coef_, index=x.columns))
print()

#NORMALIZED MODEL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model_scaled = LinearRegression().fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("------------")
print("NORMALIZED")
print("------------")
print(f"R²: {r2_scaled:.6f}")
print(f"Intercept: {model_scaled.intercept_:.4f}")
print(pd.Series(model_scaled.coef_, index=x.columns))
print()

#COMPARISON
print("------------")
print("COMPARISON")
print("------------")
print(f"R² unnormalized: {r2_raw:.6f}")
print(f"R² normalized:   {r2_scaled:.6f}")
if r2_raw == r2_scaled:
    print("\nPerformance is UNCHANGED by normalization.")
else:
    print("\nPerformance CHANGED by normalization.")

'Q7: What is the impact on the sales prediction when only radio and newspaper advertising expenditures are used as predictors?'
x_rn=df[["Radio", "Newspaper"]]

X_train_rn, X_test_rn, y_train_rn, y_test_rn = train_test_split(x_rn, y, test_size=0.2, random_state=42)

model_rn=LinearRegression().fit(X_train_rn, y_train_rn)
r2_rn=r2_score(y_test_rn, model_rn.predict(X_test_rn))

print(f"R² (all 3):          {r2_raw:.4f}")
print(f"R² (Radio + News):   {r2_rn:.4f}")
print(f"Drop:                {r2_raw - r2_rn:.4f}")

y_pred_rn=model_rn.predict(X_test_rn)
b_rn=model_rn.intercept_
coefs_rn=pd.Series(model_rn.coef_, index=x_rn.columns)

formula_rn = f"Sales = {b_rn:.3f}"
for name, value in coefs_rn.items():
    sign = "+" if value >= 0 else "-"
    formula_rn += f" {sign} {abs(value):.4f}({name})"

coef_text_rn = "\n".join(f"{n}: {v:.4f}" for n, v in coefs_rn.items())

plt.figure(figsize=(7, 5))
plt.scatter(y_test_rn, y_pred_rn, alpha=0.7, edgecolor="k")
lims=[min(y_test_rn.min(), y_pred_rn.min()),max(y_test_rn.max(), y_pred_rn.max())]
plt.plot(lims, lims, "r--", linewidth=2)
plt.text(5, 20,f"R² = {r2_rn:.3f}\n{formula_rn}\n\nCoefficients:\n{coef_text_rn}",fontsize=9,bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Radio + Newspaper Only — R² = {r2_rn:.4f}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()