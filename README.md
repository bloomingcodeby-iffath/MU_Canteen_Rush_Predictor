#  MU Canteen Rush Predictor

A machine learning-based canteen rush prediction system built from historical student footfall data. The project analyzes how time, weather, day of week, and lunch schedule affect canteen crowding — and predicts student count with rush level classification.

---

##  Project Description Summary

The goal is to **predict the number of students** visiting the canteen and classify the visit intensity into **Rush Levels (Low, Medium, High)** based on contextual features such as time slot, weather condition, day of week, and lunch timing.

The dataset was explored visually and statistically to understand distribution patterns, correlations, and behavioral trends. Two regression models — **Linear Regression** and **Random Forest Regressor** — were trained, evaluated, and compared. The final output maps predicted student counts to rush level categories using a threshold-based classification function.

---

##  Dataset

- **Source:** [MU Canteen Rush Data (GitHub)]([https://raw.githubusercontent.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor/refs/heads/main/canteen_rush_data.csv](https://raw.githubusercontent.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor/refs/heads/main/canteen_rush_data.csv))
- **Key Columns:**
  - `Time` — Time slot of canteen visit
  - `Day` — Day of the week
  - `Weather` — Weather condition (e.g., Sunny, Rainy, Cloudy)
  - `Lunch_Time` — Whether it's a designated lunch period
  - `Students` — Number of students (target variable)
  - `Rush_Level` — Categorical label: Low / Medium / High

---

##  Key Code Elements & Workflows

### 1. Exploratory Data Analysis (EDA)

#### Initial Data Inspection
Used `.head()`, `.info()`, `.isnull().sum()`, and `.unique()` to understand dataset structure, detect missing values, and examine categorical feature distributions across `Day` and `Weather`.

#### Rush Level vs Students — Bar Charts & Time vs Students-Scatter
- **Bar Plot:** Compared average student counts across Low, Medium, and High rush levels using color-coded bars (yellow → orange → red).
<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/37cb267a-8e8f-4217-acd5-98a3160d8ee9" />


#### Weather & Lunch Time Analysis
- **Box Plot:** Displayed student count spread per weather condition to detect variability and outliers.
- **Average Bar Plot:** Compared mean student footfall across weather types using `coolwarm` palette.
- **Lunch Time Bar & Pie:** Showed how designated lunch periods drive significantly higher footfall.
<img width="1489" height="530" alt="image" src="https://github.com/user-attachments/assets/75f6213f-aef5-45bf-93b2-a7e9cfab966e" />


#### Time vs Rush Level — Stacked Bar & Heatmap
- **Stacked Bar Chart:** Showed total students per time slot broken down by rush level using `viridis` colormap.
- **Heatmap:** Pivot table of Time × Rush Level with annotated student counts using `YlOrRd` color scale.
<img width="1174" height="490" alt="download" src="https://github.com/user-attachments/assets/631d5ef0-f163-47f8-b965-4e2572e19384" />


#### Pairplot by Rush Level
Used `sns.pairplot(hue='Rush_Level')` to reveal multivariate relationships between all numeric features, color-coded by rush category.
<img width="853" height="741" alt="download" src="https://github.com/user-attachments/assets/135c1962-6b00-4f32-ac1e-1438d1b71b92" />


#### Feature Correlation Heatmap
Encoded `Rush_Level` numerically (`Low=0`, `Medium=1`, `High=2`) and computed a correlation matrix for `Time`, `Lunch_Time`, `Students`, and `Rush_Level_Encoded`.
<img width="627" height="504" alt="download" src="https://github.com/user-attachments/assets/e00d1437-5a1d-420e-8f9e-bbb81e11ec24" />


---

### 2. Preprocessing

- Verified **no missing values** in the dataset.
- Applied **Label Encoding** on categorical features:
  - `Day` → `Day_Encoded`
  - `Weather` → `Weather_Encoded`
- `Rush_Level` encoded manually: `Low=0`, `Medium=1`, `High=2`

---

### 3. Feature Selection & Train-Test Split

- **Input Features (X):** `Time`, `Lunch_Time`, `Day_Encoded`, `Weather_Encoded`
- **Target Variable (y):** `Students` (continuous — regression task)
- **Split Ratio:** 80% Train / 20% Test (`random_state=42`)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training & Evaluation

### 4.1 Models Trained

| Model | Type | Library |
|---|---|---|
| Linear Regression | Parametric, Linear | `sklearn.linear_model` |
| Random Forest Regressor | Ensemble, Non-linear | `sklearn.ensemble` |

### 4.2 Training

```python
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

Random Forest used **100 decision trees** to capture non-linear patterns in student footfall data.

### 4.3 Visualizations

#### Actual vs Predicted — Scatter Plot
Both models plotted against the ideal diagonal line (`y = x`) to visually assess prediction accuracy.
<img width="841" height="624" alt="download" src="https://github.com/user-attachments/assets/fd72ff81-e69e-4bbe-bd6e-1a52513f665a" />

#### Error Distribution — Histogram
Residual distributions for both models overlaid to compare spread and bias in prediction errors.
<img width="833" height="470" alt="download" src="https://github.com/user-attachments/assets/76fc7ac7-71f8-48b4-b720-ba05aa99e669" />

#### Residual Plots
Residuals plotted against predicted values for both models to check for systematic bias or heteroscedasticity.
<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/c65dc40b-03db-4e1b-82d9-a60401dd712a" />

---

### 5. Model Evaluation Metrics

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | 13.911363 | 5.009341 |
| MSE | 292.245536 |41.552838|
| R² Score | 0.525066| 0.932472 |

<img width="695" height="451" alt="download" src="https://github.com/user-attachments/assets/093e15c9-9155-4da8-bc8a-2720ccd8bd65" />

**Random Forest** consistently outperformed Linear Regression, particularly in capturing non-linear time and weather interactions.

---

### 6. Rush Level Classification from Predictions

A threshold function maps predicted student counts to categorical rush levels:

```python
def get_rush_level(students):
    if students < 20:
        return 'Low'
    elif students < 40:
        return 'Medium'
    else:
        return 'High'
```

Applied to Random Forest predictions (best model) to produce final interpretable output.
<img width="841" height="470" alt="download" src="https://github.com/user-attachments/assets/dca0dc4b-ea55-429f-acc5-c0a2d6e529d0" />


---

##  Installation & Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project

1. Clone the repository:
```bash
git clone https://github.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor.git
cd MU_Canteen_Rush_Predictor
```

2. Run the main script:
```bash
python canteen_rush_predictor.py
```

The script will automatically load the dataset from GitHub, perform EDA, train both models, and display all visualizations sequentially.

---

## 📊 Visualization Outputs Summary

| # | Plot | Purpose |
|---|---|---|
| 1 | Rush Level Bar + Pie, Time Line + Scatter | Rush distribution overview |
| 2 | Weather Box + Bar, Lunch Bar + Pie | Feature impact on footfall |
| 3 | Stacked Bar + Heatmap | Time vs Rush Level patterns |
| 4 | Pairplot (hue=Rush_Level) | Multivariate relationship |
| 5 | Correlation Heatmap | Feature correlation strength |
| 6 | Actual vs Predicted Scatter | Model accuracy comparison |
| 7 | Error Distribution Histogram | Residual spread comparison |
| 8 | Residual Plots | Bias and variance check |
| 9 | Model Metrics Bar Chart | MAE, MSE, R² comparison |
| 10 | Final Prediction Bar Chart | Rush level output visualization |

---

##  Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Plot rendering |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Model training, evaluation, encoding |

---

## 🏆 Results

- **Best Model:** Random Forest Regressor
- **Prediction Task:** Student count regression → Rush Level classification
- **Top Rush Predictors:** Time slot and Lunch_Time were the most influential features

---

*This project demonstrates an end-to-end machine learning pipeline for real-world institutional footfall prediction, combining regression modeling with interpretable categorical output for practical canteen management.*
