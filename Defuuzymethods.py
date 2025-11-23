Max Membership Principle

# Max-Membership Principle
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 1. Universe of discourse
x = np.arange(0, 10.1, 0.1)

# 2. Trapezoidal membership function
abcd = [3, 5, 7, 9]
mfx = fuzz.trapmf(x, abcd)

print(f"Shape of x: {x.shape}")
print(f"Shape of mfx: {mfx.shape}")

# 3. Defuzzification using SOM / LOM / MOM
fom_val = fuzz.defuzz(x, mfx, 'som')   # Smallest of Maximum
lom_val = fuzz.defuzz(x, mfx, 'lom')   # Largest of Maximum
mom_val = fuzz.defuzz(x, mfx, 'mom')   # Mean of Maximum

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, mfx, 'k', linewidth=2.5, label='Fuzzy Set')

max_membership = np.max(mfx)
ax.axhline(y=max_membership, color='gray', linestyle=':', label='Max Membership')

# Heights for SOM/LOM/MOM
height_fom = fuzz.interp_membership(x, mfx, fom_val)
height_lom = fuzz.interp_membership(x, mfx, lom_val)
height_mom = fuzz.interp_membership(x, mfx, mom_val)

# Vertical lines
ax.axvline(fom_val, ymax=height_fom, color='c', linestyle='--', label=f'FoM ({fom_val:.2f})')
ax.axvline(lom_val, ymax=height_lom, color='m', linestyle='--', label=f'LoM ({lom_val:.2f})')
ax.axvline(mom_val, ymax=height_mom, color='g', linestyle='--', label=f'MoM ({mom_val:.2f})')

ax.set_title('Max-Membership Principle Defuzzification')
ax.set_xlabel('Universe of Discourse')
ax.set_ylabel('Membership Grade')
ax.legend()
ax.grid(True)

plt.show()

 Centroid 
 # Centroid Defuzzification Example
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 1. Universe of discourse
x = np.arange(0, 10.1, 0.1)

# 2. Triangular membership function
mfx = fuzz.trimf(x, [2, 5, 8])

# 3. Centroid value
centroid_value = fuzz.defuzz(x, mfx, 'centroid')
print(f"The defuzzified centroid value is: {centroid_value:.2f}")

# 4. Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, mfx, 'k', linewidth=2.5, label='Fuzzy Set')

height_centroid = fuzz.interp_membership(x, mfx, centroid_value)
ax.axvline(centroid_value, ymax=height_centroid, color='r', linestyle='--',
           label=f'Centroid ({centroid_value:.2f})')

ax.set_title('Centroid Defuzzification')
ax.set_xlabel('Universe of Discourse')
ax.set_ylabel('Membership Grade')
ax.set_ylim(-0.1, 1.1)
ax.legend()
ax.grid(True)

plt.show()

Weighted Average

# Weighted Average Defuzzification
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 1. Universe
x = np.arange(0, 11, 0.1)

# 2. Centers & Weights
centers = np.array([2, 5, 8])
weights = np.array([0.4, 0.8, 0.6])

# 3. Weighted average formula
weighted_average = np.sum(centers * weights) / np.sum(weights)
print(f"The defuzzified weighted average is: {weighted_average:.2f}")

# 4. Symmetrical triangular membership functions
low_mf = fuzz.trimf(x, [0, 2, 5])
medium_mf = fuzz.trimf(x, [2, 5, 8])
high_mf = fuzz.trimf(x, [5, 8, 10])

# 5. Visualization
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, low_mf, 'b', linewidth=1.5, label='Low')
ax.plot(x, medium_mf, 'g', linewidth=1.5, label='Medium')
ax.plot(x, high_mf, 'm', linewidth=1.5, label='High')

# Mark centers
ax.plot(centers, [1, 1, 1], 'ko', label='Centers')

# Vertical line for weighted average
height_wa = fuzz.interp_membership(x, medium_mf, weighted_average)
ax.axvline(weighted_average, ymax=height_wa, color='r', linestyle='--',
           label=f'Weighted Average ({weighted_average:.2f})')

ax.set_title('Weighted Average Defuzzification')
ax.set_xlabel('Universe of Discourse')
ax.set_ylabel('Membership Grade')
ax.set_ylim(-0.1, 1.1)
ax.legend()
ax.grid(True)

plt.show()
