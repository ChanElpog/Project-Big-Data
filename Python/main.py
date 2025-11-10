import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data directly from Excel file
# Replace 'your_file_path.xlsx' with the actual path to your Excel file
file_path = 'list.xlsx'  # เปลี่ยน 'path_to_your_file.xlsx' เป็นเส้นทางไฟล์ที่ถูกต้อง

try:
    # Read the Excel file - adjust sheet_name if needed
    df_sales_new = pd.read_excel(file_path, sheet_name=0)
    
    # Clean column names - removing spaces and special characters
    df_sales_new.columns = df_sales_new.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    # Rename columns to match our expected format
    column_mapping = {
        'Rank': 'Rank',
        'Manufacturer': 'Manufacturer',
        'Model': 'Model',
        'Form_Factor': 'Form_Factor',
        'Smartphone': 'Smartphone',
        'Year': 'Year',
        'Units_Sold_million_': 'Units_Sold'
    }
    
    # Apply mapping for columns that exist
    existing_columns = {col: column_mapping.get(col, col) for col in df_sales_new.columns if col in column_mapping}
    df_sales_new = df_sales_new.rename(columns=existing_columns)
    
    print("Successfully loaded Excel data with columns:", df_sales_new.columns.tolist())
    print(f"Total records: {len(df_sales_new)}")
    
except FileNotFoundError:
    print(f"Error: Excel file '{file_path}' not found.")
    print("Please make sure the file exists and try again.")
    # Exit or use sample data as fallback
    raise

except Exception as e:
    print(f"Error loading Excel file: {str(e)}")
    # Exit or use sample data as fallback
    raise

# 1. Data Exploration and Visualization
print("\nData Overview:")
print(df_sales_new.describe())

# Count by manufacturer
manufacturer_count = df_sales_new['Manufacturer'].value_counts()
print("\nPhone Count by Manufacturer:")
print(manufacturer_count)

# Total units sold by manufacturer
manufacturer_sales = df_sales_new.groupby('Manufacturer')['Units_Sold'].sum().sort_values(ascending=False)
print("\nTotal Units Sold by Manufacturer (millions):")
print(manufacturer_sales)

# Sales by year
yearly_sales = df_sales_new.groupby('Year')['Units_Sold'].sum()
print("\nTotal Units Sold by Year (millions):")
print(yearly_sales)

# Average sales by year
avg_yearly_sales = df_sales_new.groupby('Year')['Units_Sold'].mean()
print("\nAverage Units Sold by Year (millions):")
print(avg_yearly_sales)

# Visualize data
plt.figure(figsize=(14, 10))

# Units sold by manufacturer
plt.subplot(2, 2, 1)
top_manufacturers = manufacturer_sales.head(10)  # Top 10 manufacturers
top_manufacturers.plot(kind='bar')
plt.title('Total Units Sold by Top Manufacturers')
plt.ylabel('Units Sold (millions)')
plt.xticks(rotation=45)

# Units sold by year
plt.subplot(2, 2, 2)
yearly_sales.plot(kind='line', marker='o')
plt.title('Total Units Sold by Year')
plt.ylabel('Units Sold (millions)')
plt.grid(True)

# Average sales by manufacturer
plt.subplot(2, 2, 3)
df_sales_new.groupby('Manufacturer')['Units_Sold'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Average Units Sold per Model by Top Manufacturers')
plt.ylabel('Average Units Sold (millions)')
plt.xticks(rotation=45)

# Top 10 models by sales
plt.subplot(2, 2, 4)
top_models = df_sales_new.sort_values('Units_Sold', ascending=False).head(10)
sns.barplot(x='Units_Sold', y='Model', data=top_models)
plt.title('Top 10 Best-Selling Phone Models')
plt.xlabel('Units Sold (millions)')
plt.tight_layout()

# 2. Trend Analysis - Sales over time by top manufacturers
plt.figure(figsize=(14, 8))
top_5_manufacturers = manufacturer_sales.head(5).index.tolist()

for manufacturer in top_5_manufacturers:
    subset = df_sales_new[df_sales_new['Manufacturer'] == manufacturer]
    plt.scatter(subset['Year'], subset['Units_Sold'], label=manufacturer, alpha=0.7, s=100)
    
plt.title('Phone Sales Trends by Top 5 Manufacturers')
plt.xlabel('Year')
plt.ylabel('Units Sold (millions)')
plt.grid(True)
plt.legend()

# 3. Predictive Modeling for Future Sales

# Prepare data for modeling
X = df_sales_new[['Year', 'Manufacturer']]
y = df_sales_new['Units_Sold']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[('manufacturer', OneHotEncoder(handle_unknown='ignore'), [1])],
    remainder='passthrough'
)

# Create and train model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nModel Performance - R² (training): {train_score:.4f}")
print(f"Model Performance - R² (testing): {test_score:.4f}")

# Make predictions for future years
future_years = [2021, 2022, 2023, 2024, 2025]
top_manufacturers = manufacturer_sales.head(5).index.tolist()

# Create dataframe for predictions
future_data = []
for year in future_years:
    for manufacturer in top_manufacturers:
        future_data.append([year, manufacturer])

future_df = pd.DataFrame(future_data, columns=['Year', 'Manufacturer'])
future_predictions = model.predict(future_df)

# Add predictions to the dataframe
future_df['Predicted_Sales'] = np.maximum(0, future_predictions)  # Ensure no negative sales predictions

# Visualize predictions
plt.figure(figsize=(14, 8))

# Plot predicted sales for top manufacturers
for manufacturer in top_manufacturers:
    subset = future_df[future_df['Manufacturer'] == manufacturer]
    plt.plot(subset['Year'], subset['Predicted_Sales'], marker='o', linewidth=2, label=f'{manufacturer} (predicted)')
    
    # Plot actual data points
    actual_subset = df_sales_new[df_sales_new['Manufacturer'] == manufacturer]
    plt.scatter(actual_subset['Year'], actual_subset['Units_Sold'], s=80, alpha=0.6, label=f'{manufacturer} (actual)')

plt.title('Predicted Phone Sales by Top Manufacturers (2021-2025)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Units Sold (millions)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# 4. Market Share Analysis
plt.figure(figsize=(14, 8))

# Calculate market share by manufacturer per year
market_share = df_sales_new.groupby(['Year', 'Manufacturer'])['Units_Sold'].sum().reset_index()
yearly_totals = market_share.groupby('Year')['Units_Sold'].transform('sum')
market_share['Share'] = market_share['Units_Sold'] / yearly_totals * 100

# Focus on top manufacturers for clearer visualization
top_manufacturers_share = market_share[market_share['Manufacturer'].isin(top_manufacturers)]

# Plot market share trends
pivot_share = top_manufacturers_share.pivot(index='Year', columns='Manufacturer', values='Share')
pivot_share.plot(kind='area', stacked=True, alpha=0.7, colormap='viridis')
plt.title('Market Share Trends by Top Manufacturers', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Market Share (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title='Manufacturer', fontsize=12)

# 5. Advanced Analysis - Correlation between release year and sales performance
plt.figure(figsize=(12, 7))
sns.boxplot(x='Year', y='Units_Sold', data=df_sales_new)
plt.title('Sales Distribution by Release Year', fontsize=16)
plt.xlabel('Release Year', fontsize=14)
plt.ylabel('Units Sold (millions)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# 6. Feature Importance Analysis for Sales using Random Forest
# We'll include additional features if available

# Determine features to use based on available columns
potential_features = ['Manufacturer', 'Year']
if 'Form_Factor' in df_sales_new.columns:
    potential_features.append('Form_Factor')
if 'Smartphone' in df_sales_new.columns:
    potential_features.append('Smartphone')

# Create feature matrix
X_features = pd.get_dummies(df_sales_new[potential_features], drop_first=True)
y_sales = df_sales_new['Units_Sold']

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_features, y_sales)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_features.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top Features Affecting Phone Sales', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()

# 7. Future trends analysis - insights summary
top_mfr_trend = future_df.pivot(index='Year', columns='Manufacturer', values='Predicted_Sales')
percent_changes = ((top_mfr_trend.iloc[-1] / top_mfr_trend.iloc[0]) - 1) * 100

print("\nKey Insights and Future Trends:")
print(f"1. Projected 5-year growth by manufacturer (2021-2025):")
for mfr, change in percent_changes.items():
    print(f"   - {mfr}: {change:.1f}% {'increase' if change > 0 else 'decrease'}")

print("\n2. Top selling models overall:")
for idx, row in top_models.head(5).iterrows():
    print(f"   - {row['Model']} by {row['Manufacturer']}: {row['Units_Sold']} million units")

# 8. Save results to Excel
results_file = 'mobile_phone_sales_analysis_results.xlsx'

with pd.ExcelWriter(results_file) as writer:
    df_sales_new.to_excel(writer, sheet_name='Original_Data', index=False)
    future_df.to_excel(writer, sheet_name='Future_Predictions', index=False)
    feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
    manufacturer_sales.to_excel(writer, sheet_name='Manufacturer_Totals')
    yearly_sales.to_excel(writer, sheet_name='Yearly_Totals')
    
print(f"\nAnalysis results saved to {results_file}")

plt.show()
