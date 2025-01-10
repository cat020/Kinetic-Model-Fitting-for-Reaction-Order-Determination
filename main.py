import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from flask import Flask, request, send_file, render_template
import io

app = Flask(__name__)

# Function to calculate R-squared
def calculate_r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

# Zero-order reaction: C_A = C_A0 - kt
def zero_order_model(t, C_A0, k):
    return C_A0 - k * t

# First-order reaction: C_A = C_A0 * exp(-kt)
def first_order_model(t, C_A0, k):
    return C_A0 * np.exp(-k * t)

# Second-order reaction: 1/C_A = 1/C_A0 + kt
def second_order_model(t, C_A0, k):
    return 1 / (1 / C_A0 + k * t)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    # Get data from the form
    time_str = request.form['time']
    concentration_str = request.form['concentration']
    order = request.form['order']

    # Convert comma-separated strings to numpy arrays
    time = np.array([float(x) for x in time_str.split(',')])
    concentration = np.array([float(x) for x in concentration_str.split(',')])

    # Perform curve fitting and linear regression based on the selected order
    if order == 'zero':
        popt, _ = curve_fit(zero_order_model, time, concentration, p0=[10, 0.1])
        C_A0, k = popt
        fitted_concentration_curve = zero_order_model(time, *popt)
        r_squared_curve = calculate_r_squared(concentration, fitted_concentration_curve)

        slope, intercept, r_value, _, _ = linregress(time, concentration)
        fitted_concentration_linear = intercept + slope * time
        r_squared_linear = r_value**2

    elif order == 'first':
        popt, _ = curve_fit(first_order_model, time, concentration, p0=[10, 0.01])
        C_A0, k = popt
        fitted_concentration_curve = first_order_model(time, *popt)
        r_squared_curve = calculate_r_squared(concentration, fitted_concentration_curve)

        ln_concentration = np.log(concentration)
        slope, intercept, r_value, _, _ = linregress(time, ln_concentration)
        fitted_concentration_linear = np.exp(intercept + slope * time)
        r_squared_linear = r_value**2

    elif order == 'second':
        popt, _ = curve_fit(second_order_model, time, concentration, p0=[10, 0.01])
        C_A0, k = popt
        fitted_concentration_curve = second_order_model(time, *popt)
        r_squared_curve = calculate_r_squared(concentration, fitted_concentration_curve)

        inv_concentration = 1 / concentration
        slope, intercept, r_value, _, _ = linregress(time, inv_concentration)
        fitted_concentration_linear = 1 / (intercept + slope * time)
        r_squared_linear = r_value**2

    else:
        return "Unsupported order", 400

    # Plot the data and the fits
    plt.figure(figsize=(12, 6))
    plt.plot(time, concentration, 'o', label='Data')
    plt.plot(time, fitted_concentration_curve, 'r', label=f'{order.capitalize()}-order Curve Fit: R²={r_squared_curve:.4f}')
    plt.plot(time, fitted_concentration_linear, 'g--', label=f'{order.capitalize()}-order Linear Fit: R²={r_squared_linear:.4f}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Concentration (mol/liter)')
    plt.title(f'{order.capitalize()}-order Reaction')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
