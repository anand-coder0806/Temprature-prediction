import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def main(args):
	print(f"Loading dataset from {args.data}...")
	df = pd.read_csv(args.data)

	print("\n--- Exploratory Data Analysis ---")
	print("Missing values in each column:")
	print(df.isnull().sum())
	print("\nSummary statistics:")
	print(df.describe())

	print("\nGenerating scatter plot...")
	plt.figure(figsize=(10, 6))
	sample_df = df.sample(n=min(args.sample_plot_points, len(df)), random_state=42)
	sns.scatterplot(x='humidity', y='temperature', data=sample_df, alpha=0.5)
	plt.title('Humidity vs Temperature (Sampled)')
	plt.xlabel('Humidity')
	plt.ylabel('Temperature')
	plt.savefig('humidity_vs_temperature.png')
	plt.close()
	print("Saved scatter plot as 'humidity_vs_temperature.png'")

	print("\n--- Data Preprocessing ---")
	df = df.dropna(subset=['humidity', 'temperature'])
	X = df[['humidity']]
	y = df['temperature']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=42
	)
	print(f"Training set size: {X_train.shape[0]}")
	print(f"Testing set size: {X_test.shape[0]}")

	print(f"\n--- Model Training (Polynomial Regression, degree={args.degree}) ---")
	poly = PolynomialFeatures(degree=args.degree)
	X_train_poly = poly.fit_transform(X_train)
	X_test_poly = poly.transform(X_test)

	model = LinearRegression()
	model.fit(X_train_poly, y_train)

	y_pred = model.predict(X_test_poly)
	print("Model training and prediction complete.")

	print("\n--- Evaluation ---")
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error: {mse:.4f}")
	print(f"R-squared: {r2:.4f}")

	print("\nGenerating regression line plot...")
	plt.figure(figsize=(10, 6))
	n_samples = min(args.sample_plot_points, len(X_test))
	if n_samples == 0:
		print("Not enough test samples to generate regression plot.")
		return

	rng = np.random.default_rng(42)
	test_sample_indices = rng.choice(len(X_test), n_samples, replace=False)
	X_test_sample = X_test.iloc[test_sample_indices]
	y_test_sample = y_test.iloc[test_sample_indices]
	y_pred_sample = y_pred[test_sample_indices]

	# Sort the values so the polynomial line draws smoothly
	sorted_zip = sorted(zip(X_test_sample['humidity'], y_pred_sample))
	x_plot, y_plot = zip(*sorted_zip)

	plt.scatter(X_test_sample['humidity'], y_test_sample, color='blue', alpha=0.3, label='Actual Data')
	plt.plot(x_plot, y_plot, color='red', linewidth=3, label=f'Polynomial Curve (Degree {args.degree})')
	plt.title('Polynomial Regression: Humidity vs Temperature')
	plt.xlabel('Humidity')
	plt.ylabel('Temperature')
	plt.legend()
	plt.savefig('regression_line.png')
	plt.close()
	print("Saved regression line plot as 'regression_line.png'")
	if args.save_model:
		model_bundle = {
			'poly': poly,
			'model': model,
		}
		joblib.dump(model_bundle, args.save_model)
		print(f"Saved trained model bundle to {args.save_model}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a polynomial regression on humidity->temperature')
	parser.add_argument('--data', type=str, default='humidity.csv', help='Path to CSV data file')
	parser.add_argument('--degree', type=int, default=3, help='Polynomial degree')
	parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
	parser.add_argument('--sample-plot-points', type=int, default=5000, help='Max points to sample for plots')
	parser.add_argument('--save-model', type=str, default='', help='Path to save trained model bundle (joblib)')
	args = parser.parse_args()
	main(args)
