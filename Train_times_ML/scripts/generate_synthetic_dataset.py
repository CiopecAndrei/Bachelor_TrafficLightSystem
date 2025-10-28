import pandas as pd
import random
import os
from collections import defaultdict

# Settings
SAMPLES_PER_BUCKET_PER_PHASE = 50  # 50 for each of 3 buckets, for both SN and EW
MAX_CARS = 3
random.seed(42)

data = []
bucket_counts = {
    'SN': defaultdict(int),
    'EW': defaultdict(int)
}

tie_counter = 0  # To alternate tie assignments

def assign_bucket(total_cars):
    if total_cars <= 2:
        return 0  # Short
    elif total_cars <= 4:
        return 1  # Medium
    else:
        return 2  # Long

# Generate balanced samples
while any(bucket_counts[phase][bucket] < SAMPLES_PER_BUCKET_PER_PHASE
          for phase in ['SN', 'EW'] for bucket in range(3)):

    # Random lane counts
    cars_south = random.randint(0, MAX_CARS)
    cars_north = random.randint(0, MAX_CARS)
    cars_west = random.randint(0, MAX_CARS)
    cars_east = random.randint(0, MAX_CARS)

    sn_total = cars_south + cars_north
    ew_total = cars_east + cars_west

    # Determine dominant direction (balanced tie handling)
    if sn_total > ew_total:
        light_phase = 'SN'
        active_cars = sn_total
    elif ew_total > sn_total:
        light_phase = 'EW'
        active_cars = ew_total
    else:
        # Alternate tie between SN and EW
        light_phase = 'SN' if tie_counter % 2 == 0 else 'EW'
        active_cars = sn_total  # same as ew_total
        tie_counter += 1

    green_time_bucket = assign_bucket(active_cars)

    # Check if this bucket still needs more samples
    if bucket_counts[light_phase][green_time_bucket] < SAMPLES_PER_BUCKET_PER_PHASE:
        data.append({
            'cars_south': cars_south,
            'cars_north': cars_north,
            'cars_west': cars_west,
            'cars_east': cars_east,
            'light_phase': light_phase,
            'green_time_bucket': green_time_bucket
        })
        bucket_counts[light_phase][green_time_bucket] += 1

# DataFrame and stats
df = pd.DataFrame(data)

print("Dataset size:", len(df))
print("\nGreen phase distribution:")
print(df['light_phase'].value_counts())
print("\nGreen time bucket distribution per phase:")
print(df.groupby(['light_phase', 'green_time_bucket']).size())

# Save
output_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\data\traffic_timing_balanced.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
