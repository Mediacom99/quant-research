# Exploratory Data Analysis of cleaned data


# Scale the data to have mean 0 and std 1
scaler = StandardScaler()
scaled_data = pd.Series(scaler.fit_transform(series.to_numpy().reshape(-1, 1)).flatten())

scaled_data.plot(kind='hist', bins = 500, density=True)
mu,sigma = norm.fit(scaled_data)

print(mu,sigma)

# Plot fitted Gaussian
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
print(x)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k',linewidth=2, label=f'Fitted Gaussian\n$\mu$={mu:.2f}, $\sigma$={sigma:.2f}')

## TESTING FIT
fitted_samples = norm.rvs(loc=mu, scale=sigma, size=len(scaled_data))
t_statistic, p_value = ttest_ind(scaled_data, fitted_samples)

print(f"t-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")   


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



