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


# --- Get ID lists from the personas DataFrame ---
honeytrap_ids = list(personas_df[personas_df["is_honeytrap"] == True]["persona_id"])
legit_ids = list(personas_df[personas_df["is_honeytrap"] == False]["persona_id"])
# Ensure high_profile_ids are correctly identified from the generated personas
high_profile_df = personas_df[personas_df['profile_type'].isin(['Military', 'Executive'])]
high_profile_ids = list(high_profile_df['persona_id'])


# --- Generate Interactions ---
print("\nGenerating interactions...")
interactions_data = []
interaction_counter = 1

# 1. Legitimate User Interactions (Random and sporadic)
for source_id in legit_ids:
    num_interactions = random.randint(10, 60)
    for _ in range(num_interactions):
        possible_targets = [id for id in legit_ids if id != source_id]
        target_id = random.choice(possible_targets)

        interactions_data.append({
            "interaction_id": interaction_counter,
            "source_persona_id": source_id,
            "target_persona_id": target_id,
            "timestamp": fake.date_time_between(start_date="-1y", end_date="now"),
            "has_link": np.random.choice([True, False], p=[0.05, 0.95]), # Low link sharing
            "has_emoji": np.random.choice([True, False], p=[0.4, 0.6]),
        })
        interaction_counter += 1

# 2. Honeytrap Interactions (Two-phase: Grooming -> Exploitation)
for source_id in honeytrap_ids:
    num_targets = random.randint(2, 5)
    targets_for_this_attacker = random.sample(high_profile_ids, num_targets)

    for target_id in targets_for_this_attacker:
        grooming_start_time = fake.date_time_between(start_date="-3m", end_date="-1m")
        
        # --- Grooming Phase: Build trust ---
        num_grooming_messages = random.randint(4, 8)
        for i in range(num_grooming_messages):
            interactions_data.append({
                "interaction_id": interaction_counter,
                "source_persona_id": source_id,
                "target_persona_id": target_id,
                "timestamp": grooming_start_time + timedelta(days=i*random.randint(1,3)), # Spread out
                "has_link": False, # No links during grooming
                "has_emoji": np.random.choice([True, False], p=[0.7, 0.3]), # Higher emoji use for flattery
            })
            interaction_counter += 1
            
        # --- Exploitation Phase: The attack ---
        exploitation_start_time = fake.date_time_between(start_date="-1w", end_date="now")
        num_exploit_messages = random.randint(8, 20)
        for i in range(num_exploit_messages):
             interactions_data.append({
                "interaction_id": interaction_counter,
                "source_persona_id": source_id,
                "target_persona_id": target_id,
                "timestamp": exploitation_start_time + timedelta(minutes=i*random.randint(1,10)), # Intense burst
                "has_link": np.random.choice([True, False], p=[0.35, 0.65]), # High chance of sending links
                "has_emoji": np.random.choice([True, False], p=[0.2, 0.8]), # More direct, less flattery
            })
             interaction_counter += 1

# Create DataFrame
interactions_df = pd.DataFrame(interactions_data)

# Save to CSV
interactions_df.to_csv("interactions.csv", index=False)

print("✅ interactions.csv generated successfully.")
print(interactions_df.tail()) # Show tail to see attacker interactions


# --- Define Honeytrap Campaign Infrastructure ---
print("\nGenerating infrastructure data...")
# Group honeytraps into 10 campaign cells of 5 members each
campaign_map = {pid: f"cell_{i//5}" for i, pid in enumerate(honeytrap_ids)}
malicious_domains = [f"verify-secure-login-part-{i}.io" for i in range(10)]
benign_domains = ["google.com", "linkedin.com", "github.com", "forbes.com"]

# Each cell gets its own pool of shared infrastructure to rotate through
campaign_infra = {
    f"cell_{i}": {
        "ip_pool": [fake.ipv4() for _ in range(5)], # Pool of 5 IPs per cell
        "device_pool": [fake.android_platform_token() for _ in range(3)], # Pool of 3 devices
        "domain": malicious_domains[i]
    } for i in range(10)
}

# --- Generate Infrastructure Data based on Interactions ---
infra_data = []
legit_infra_cache = {} # Cache legit IPs/devices for consistency

for _, row in interactions_df.iterrows():
    source_id = row["source_persona_id"]
    infra_info = {"persona_id": source_id, "interaction_id": row["interaction_id"]}

    if source_id in honeytrap_ids:
        campaign_id = campaign_map[source_id]
        # Randomly pick from the campaign's pool to simulate rotation
        infra_info["ip_address"] = random.choice(campaign_infra[campaign_id]["ip_pool"])
        infra_info["device_id"] = random.choice(campaign_infra[campaign_id]["device_pool"])
        infra_info["domain_used"] = campaign_infra[campaign_id]["domain"] if row["has_link"] else None
    else: # Legitimate user
        if source_id not in legit_infra_cache:
            legit_infra_cache[source_id] = {"ip": fake.ipv4(), "device_id": fake.android_platform_token()}
        
        infra_info["ip_address"] = legit_infra_cache[source_id]["ip"]
        infra_info["device_id"] = legit_infra_cache[source_id]["device_id"]
        infra_info["domain_used"] = random.choice(benign_domains) if row["has_link"] else None

    infra_data.append(infra_info)

# Create DataFrame
infrastructure_df = pd.DataFrame(infra_data)

# Save to CSV
infrastructure_df.to_csv("infrastructure.csv", index=False)

print("✅ infrastructure.csv generated successfully.")
print(infrastructure_df.tail()) # Show tail to see attacker infrastructure