import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Load saved model and make predictions')
    parser.add_argument('--model', type=str, default='model.joblib', help='Path to saved model bundle')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--value', type=float, help='Single humidity value to predict temperature for')
    group.add_argument('--input-csv', type=str, help='CSV file with a `humidity` column to predict for')
    parser.add_argument('--output-csv', type=str, help='If set with --input-csv, save predictions to this CSV')
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    poly = bundle['poly']
    model = bundle['model']

    if args.value is not None:
        X = [[args.value]]
        X_poly = poly.transform(X)
        y_pred = model.predict(X_poly)
        print(f"Predicted temperature for humidity={args.value}: {y_pred[0]:.4f}")
        return

    df = pd.read_csv(args.input_csv)
    if 'humidity' not in df.columns:
        raise SystemExit("Input CSV must contain a 'humidity' column")

    X = df[['humidity']]
    X_poly = poly.transform(X)
    preds = model.predict(X_poly)
    df['predicted_temperature'] = preds

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
    else:
        print(df[['humidity', 'predicted_temperature']].head())


if __name__ == '__main__':
    main()
