import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define parameters
num_patients = 1000
num_simulations = 500
baseline_mean = 60
baseline_std = 15

# Generate baseline waiting times
baseline_waiting_times = np.random.normal(baseline_mean, baseline_std, num_patients)

# Define urgent/non-urgent classification
np.random.seed(42)
symptom_severity = np.random.uniform(0, 1, num_patients)
urgent_patients = symptom_severity > 0.7
non_urgent_patients = ~urgent_patients

# Define reduction percentages
urgent_reduction = 0.60
non_urgent_reduction = 0.30

# Monte Carlo simulation to generate `simulated_waiting_times`
simulated_waiting_times = []
for _ in range(num_simulations):
    # Simulate variability in triage effectiveness (±5%)
    urgent_variation = urgent_reduction + np.random.uniform(-0.05, 0.05)
    non_urgent_variation = non_urgent_reduction + np.random.uniform(-0.05, 0.05)
    
    simulated_times = np.copy(baseline_waiting_times)
    simulated_times[urgent_patients] *= (1 - urgent_variation)
    simulated_times[non_urgent_patients] *= (1 - non_urgent_variation)
    
    simulated_waiting_times.append(simulated_times.mean())

# Bootstrapping to calculate confidence intervals
bootstrapped_means = []
for _ in range(1000):  # Number of bootstrap samples
    bootstrap_sample = np.random.choice(simulated_waiting_times, size=num_simulations, replace=True)
    bootstrapped_means.append(np.mean(bootstrap_sample))

# Calculate 95% confidence interval
ci_lower = np.percentile(bootstrapped_means, 2.5)
ci_upper = np.percentile(bootstrapped_means, 97.5)

# Visualization of the bootstrap results
plt.figure(figsize=(12, 6))
sns.histplot(bootstrapped_means, kde=True, bins=30, color='green', alpha=0.7)
plt.axvline(ci_lower, color='red', linestyle='dashed', linewidth=1, label=f"2.5% CI: {ci_lower:.2f} mins")
plt.axvline(ci_upper, color='blue', linestyle='dashed', linewidth=1, label=f"97.5% CI: {ci_upper:.2f} mins")
plt.axvline(np.mean(bootstrapped_means), color='purple', linestyle='solid', linewidth=2, label=f"Mean: {np.mean(bootstrapped_means):.2f} mins")
plt.title("Bootstrapping Results: Average Waiting Times (Urgent/Non-Urgent)")
plt.xlabel("Average Waiting Time (Minutes)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Print the confidence interval
print(f"Bootstrapped 95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f}) minutes")

