import argparse
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(description='Train with hyperparameter tuning for polynomial degree')
    parser.add_argument('--data', type=str, default='humidity.csv')
    parser.add_argument('--degrees', type=int, nargs='+', default=[1,2,3,4,5])
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--save-model', type=str, default='model_tuned.joblib')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=['humidity', 'temperature'])
    X = df[['humidity']]
    y = df['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('lr', LinearRegression())
    ])

    param_grid = {
        'poly__degree': args.degrees
    }

    grid = GridSearchCV(pipeline, param_grid, cv=args.cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f'Best degree: {grid.best_params_["poly__degree"]}')

    y_pred = best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}, R2: {r2:.4f}')

    # Save bundle compatible with predict.py/api.py
    bundle = {'poly': best.named_steps['poly'], 'model': best.named_steps['lr']}
    joblib.dump(bundle, args.save_model)
    print(f'Saved tuned model to {args.save_model}')


if __name__ == '__main__':
    main()
