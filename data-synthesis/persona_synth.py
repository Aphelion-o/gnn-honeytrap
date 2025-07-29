import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for generating fake data
fake = Faker()

# --- Configuration ---
TOTAL_PERSONAS = 1000
HONEYTRAP_COUNT = 50
HIGH_PROFILE_COUNT = 200

# --- Generate Personas ---
print("Generating personas with randomized honeytrap IDs...")

# 1. Create a pool of all possible persona IDs from 1 to 1000
all_persona_ids = list(range(1, TOTAL_PERSONAS + 1))

# 2. Randomly sample from the pool to decide which IDs will be honeytraps
#    This is the key change to eliminate bias.
honeytrap_ids_set = set(random.sample(all_persona_ids, HONEYTRAP_COUNT))

# 3. Identify the legitimate and high-profile IDs from the remaining pool
legit_ids = [pid for pid in all_persona_ids if pid not in honeytrap_ids_set]
high_profile_ids_set = set(random.sample(legit_ids, HIGH_PROFILE_COUNT))

personas_data = []
# 4. Loop through all IDs and assign attributes based on whether they are in our sets
for i in all_persona_ids:
    persona_info = {"persona_id": i}

    # Check if the current ID was randomly chosen to be a honeytrap
    if i in honeytrap_ids_set:
        persona_info["is_honeytrap"] = True
        # 80% of honeytraps are new accounts, 20% are older "sleeper" accounts
        if random.random() < 0.8:
            persona_info["creation_date"] = fake.date_time_between(start_date="-3m", end_date="now")
        else:
            persona_info["creation_date"] = fake.date_time_between(start_date="-2y", end_date="-1y")
        persona_info["profile_type"] = "Standard"
    else: # This is a legitimate user
        persona_info["is_honeytrap"] = False
        persona_info["creation_date"] = fake.date_time_between(start_date="-5y", end_date="-4m")
        # Assign high-profile or civilian roles
        if i in high_profile_ids_set:
            persona_info["profile_type"] = random.choice(["Military", "Executive"])
        else:
            persona_info["profile_type"] = "Civilian"

    personas_data.append(persona_info)

# Create DataFrame
personas_df = pd.DataFrame(personas_data)

# Save to CSV
personas_df.to_csv("personas.csv", index=False)

print("✅ personas.csv generated successfully with randomized honeytraps.")
print(personas_df.head())
# Verify by checking a few known honeytrap IDs
sample_honeytrap_id = list(honeytrap_ids_set)[0]
print("\nSample honeytrap profile:")
print(personas_df[personas_df['persona_id'] == sample_honeytrap_id])