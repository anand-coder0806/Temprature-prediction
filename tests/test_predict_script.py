import subprocess
import sys
import tempfile
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np


def test_predict_script_single_value():
    # Create a tiny model bundle and save it
    poly = PolynomialFeatures(degree=1)
    X = np.array([[0.], [10.], [20.], [30.]])
    Xp = poly.fit_transform(X)
    y = np.array([0., 5., 10., 15.])
    lr = LinearRegression().fit(Xp, y)
    bundle = {'poly': poly, 'model': lr}

    tmp = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
    joblib.dump(bundle, tmp.name)

    # Run predict.py
    proc = subprocess.run([sys.executable, 'predict.py', '--model', tmp.name, '--value', '15'], capture_output=True, text=True)
    assert proc.returncode == 0
    assert 'Predicted temperature' in proc.stdout
