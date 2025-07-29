import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import emoji

# Load MobileBERT (or Sentence-BERT if you switch models)
model_name = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()


def embed_text(text):
    """Generate embeddings for text using MobileBERT"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Use mean pooling of last hidden state
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()


def has_emoji(text):
    """Check if text contains emoji"""
    return bool(emoji.get_emoji_regexp().search(text))


def contains_media_indicators(text):
    """Check if message contains media indicators"""
    media_patterns = [
        r'\[image\]', r'\[photo\]', r'\[video\]', r'\[audio\]', r'\[file\]',
        r'<image>', r'<photo>', r'<video>', r'<audio>', r'<file>',
        r'sent a photo', r'sent an image', r'sent a video', r'sent a file',
        r'\.jpg', r'\.jpeg', r'\.png', r'\.gif', r'\.mp4', r'\.mov', r'\.pdf'
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in media_patterns)


def contains_link(text):
    """Check if text contains URLs"""
    url_patterns = [
        r'http[s]?://',
        r'www\.',
        r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',
        r'bit\.ly',
        r'tinyurl',
        r't\.co/'
    ]
    return any(re.search(pattern, text) for pattern in url_patterns)


def calculate_topic_diversity(embeddings, min_clusters=2, max_clusters=10):
    """Calculate topic diversity using KMeans clustering on embeddings"""
    if len(embeddings) < min_clusters:
        return 0.0
    
    embeddings_array = np.array(embeddings)
    n_clusters = min(max_clusters, len(embeddings))
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        unique_clusters = len(set(cluster_labels))
        return unique_clusters / len(embeddings)
    except:
        return 0.0


def calculate_message_frequency_stability(timestamps):
    """Calculate entropy/variance of message time gaps"""
    if len(timestamps) < 2:
        return 0.0
    
    # Sort timestamps and calculate gaps in seconds
    sorted_times = sorted(timestamps)
    gaps = [(t2 - t1).total_seconds() for t1, t2 in zip(sorted_times, sorted_times[1:])]
    
    if not gaps:
        return 0.0
    
    # Use standard deviation as stability measure (higher = less stable)
    return np.std(gaps)


def parse_conversation(convo_data, verified_users=None):
    """
    Parse conversation data and extract user features and message records
    
    Args:
        convo_data: List of message dictionaries with keys: timestamp, sender, receiver, text
        verified_users: Set of user IDs that are considered verified/trusted (optional)
    
    Returns:
        user_features: Dictionary of user IDs to feature dictionaries
        message_records: List of message records with embeddings
    """
    
    if verified_users is None:
        verified_users = set()
    
    user_stats = defaultdict(lambda: {
        "messages_sent": 0,
        "messages_received": 0,
        "flirt_words": 0,
        "links_sent": 0,
        "sensitive_words": 0,
        "emoji_messages": 0,
        "media_messages": 0,
        "timestamps": [],
        "receivers": set(),
        "senders": set(),
        "message_lengths": [],
        "embeddings": [],
        "verified_interactions": 0,
        "total_interactions": 0,
    })

    # Define keyword sets for feature extraction
    flirt_keywords = {
        "babe", "sweet", "hot", "gorgeous", "cute", "handsome", "love", "beautiful", 
        "sexy", "amazing", "incredible", "stunning", "attractive", "charming", "cutie",
        "honey", "darling", "sweetheart", "angel", "perfect", "divine", "breathtaking"
    }
    
    sensitive_keywords = {
        "money", "help", "emergency", "hospital", "visa", "transfer", "payment", 
        "cash", "bank", "account", "financial", "urgent", "crisis", "loan", "debt",
        "invest", "bitcoin", "crypto", "wire", "western union", "paypal", "venmo",
        "inheritance", "lawyer", "legal", "documents", "passport", "immigration"
    }

    message_records = []

    print(f"Processing {len(convo_data)} messages...")
    
    # First pass: collect all messages and basic stats
    for i, entry in enumerate(convo_data):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(convo_data)} messages")
            
        sender = entry["sender"]
        receiver = entry["receiver"]
        text = entry["text"]
        text_lower = text.lower()
        
        # Parse timestamp - handle both with and without timezone
        timestamp_str = entry["timestamp"]
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        timestamp = datetime.fromisoformat(timestamp_str)

        # Generate embedding for the message
        embedding = embed_text(text)
        
        # Update sender statistics
        user_stats[sender]["messages_sent"] += 1
        user_stats[sender]["timestamps"].append(timestamp)
        user_stats[sender]["receivers"].add(receiver)
        user_stats[sender]["message_lengths"].append(len(text))
        user_stats[sender]["embeddings"].append(embedding)
        user_stats[sender]["total_interactions"] += 1
        
        # Update receiver statistics
        user_stats[receiver]["messages_received"] += 1
        user_stats[receiver]["senders"].add(sender)
        user_stats[receiver]["total_interactions"] += 1

        # Check verified interactions
        if receiver in verified_users:
            user_stats[sender]["verified_interactions"] += 1
        if sender in verified_users:
            user_stats[receiver]["verified_interactions"] += 1

        # Count keyword occurrences for sender
        if any(word in text_lower for word in flirt_keywords):
            user_stats[sender]["flirt_words"] += 1
        if any(word in text_lower for word in sensitive_keywords):
            user_stats[sender]["sensitive_words"] += 1
        if contains_link(text):
            user_stats[sender]["links_sent"] += 1
        if has_emoji(text):
            user_stats[sender]["emoji_messages"] += 1
        if contains_media_indicators(text):
            user_stats[sender]["media_messages"] += 1
        
        # Create message record
        message_record = {
            "id": f"msg_{i:06d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "text": text_lower,  # Store lowercase version
            "timestamp": timestamp.isoformat(),
            "embedding": embedding.tolist(),
        }
        message_records.append(message_record)

    print(f"Completed processing all messages. Calculating user features...")

    # Calculate user features
    user_features = {}
    for user, stats in user_stats.items():
        times = sorted(stats["timestamps"])
        total_msgs_sent = stats["messages_sent"]
        total_msgs_received = stats["messages_received"]
        
        # Basic validation
        if total_msgs_sent == 0:
            # User only received messages, set default values
            user_features[user] = {
                "interaction_frequency": 0.0,
                "bidirectional_ratio": 0.0,
                "conversation_duration_months": 0.0,
                "message_frequency_stability": 0.0,
                "is_verified_interaction": stats["verified_interactions"] / max(1, stats["total_interactions"]),
                "link_ratio": 0.0,
                "sensitive_word_ratio": 0.0,
                "flirt_or_bait_ratio": 0.0,
                "suspicious_ratio": 0.0,
                "topic_diversity": 0.0,
                "emoji_use_ratio": 0.0,
                "media_sent_ratio": 0.0,
                "avg_message_length": 0.0,
                "num_unique_partners": len(stats["senders"]),
            }
            continue
        
        # Calculate time-based features
        if len(times) > 1:
            time_span = (times[-1] - times[0]).total_seconds()
            time_span_days = max(1, time_span / (24 * 3600))  # Convert to days, minimum 1
            conversation_duration_months = time_span_days / 30.0
            message_frequency_stability = calculate_message_frequency_stability(times)
        else:
            time_span_days = 1
            conversation_duration_months = 1.0 / 30.0  # Assume 1 day for single message
            message_frequency_stability = 0.0

        # Calculate topic diversity
        topic_diversity = calculate_topic_diversity(stats["embeddings"])
        
        # Calculate ratios
        interaction_frequency = total_msgs_sent / time_span_days
        bidirectional_ratio = total_msgs_received / total_msgs_sent if total_msgs_sent > 0 else 0.0
        is_verified_interaction = stats["verified_interactions"] / max(1, stats["total_interactions"])
        link_ratio = stats["links_sent"] / total_msgs_sent
        sensitive_word_ratio = stats["sensitive_words"] / total_msgs_sent
        flirt_or_bait_ratio = stats["flirt_words"] / total_msgs_sent
        emoji_use_ratio = stats["emoji_messages"] / total_msgs_sent
        media_sent_ratio = stats["media_messages"] / total_msgs_sent
        
        # Calculate suspicious ratio (heuristic combination)
        # High link + high flirty + high initiation (low bidirectional) + low verification
        suspicious_components = [
            min(1.0, link_ratio * 2),  # Links weighted heavily
            min(1.0, flirt_or_bait_ratio * 1.5),  # Flirty messages
            min(1.0, (1 - bidirectional_ratio)),  # One-sided conversations
            min(1.0, (1 - is_verified_interaction)),  # Unverified interactions
            min(1.0, sensitive_word_ratio * 3),  # Sensitive words weighted heavily
        ]
        suspicious_ratio = np.mean(suspicious_components)

        # Store all features
        user_features[user] = {
            "interaction_frequency": interaction_frequency,
            "bidirectional_ratio": bidirectional_ratio,
            "conversation_duration_months": conversation_duration_months,
            "message_frequency_stability": message_frequency_stability,
            "is_verified_interaction": is_verified_interaction,
            "link_ratio": link_ratio,
            "sensitive_word_ratio": sensitive_word_ratio,
            "flirt_or_bait_ratio": flirt_or_bait_ratio,
            "suspicious_ratio": suspicious_ratio,
            "topic_diversity": topic_diversity,
            "emoji_use_ratio": emoji_use_ratio,
            "media_sent_ratio": media_sent_ratio,
            "avg_message_length": np.mean(stats["message_lengths"]) if stats["message_lengths"] else 0.0,
            "num_unique_partners": len(stats["receivers"]),
        }

    print(f"Generated features for {len(user_features)} users")
    return user_features, message_records


def save_processed_data(user_features, message_records, output_dir="processed"):
    """Save the processed data to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save user features
    user_features_path = os.path.join(output_dir, "user_features.json")
    with open(user_features_path, "w") as f:
        json.dump(user_features, f, indent=2)
    
    # Save message records
    message_records_path = os.path.join(output_dir, "message_records.json")
    with open(message_records_path, "w") as f:
        json.dump(message_records, f, indent=2)
    
    print(f"Saved user features to: {user_features_path}")
    print(f"Saved message records to: {message_records_path}")
    
    # Print some statistics
    print(f"\nData Statistics:")
    print(f"  Total users: {len(user_features)}")
    print(f"  Total messages: {len(message_records)}")
    print(f"  Average messages per user: {len(message_records) / len(user_features):.2f}")
    
    # Show sample of user features
    print(f"\nSample user features:")
    for i, (user_id, features) in enumerate(list(user_features.items())[:2]):
        print(f"  {user_id}:")
        for key, value in features.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
        if i < 1:
            print()

    # Show feature distribution summary
    print(f"\nFeature Distribution Summary:")
    feature_stats = defaultdict(list)
    for user_features_dict in user_features.values():
        for feature_name, value in user_features_dict.items():
            if isinstance(value, (int, float)):
                feature_stats[feature_name].append(value)
    
    for feature_name, values in feature_stats.items():
        if values:
            print(f"  {feature_name}:")
            print(f"    Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
            print(f"    Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")


def load_sample_data():
    """Load and return sample conversation data in the expected format"""
    sample_data = [
        {
            "timestamp": "2025-07-29T21:15:12Z",
            "sender": "user_001",
            "receiver": "user_002",
            "text": "Hey there. Your profile mentioned you're into hiking. That picture from the mountaintop looks incredible. I've always wanted to go somewhere like that."
        },
        {
            "timestamp": "2025-07-29T21:28:45Z",
            "sender": "user_002",
            "receiver": "user_001",
            "text": "Hey! Thanks, that was from a trip last year. It was a tough climb but totally worth it for the view. You should definitely try it sometime. Your photos are really nice too, you have a great eye. 😊"
        },
        {
            "timestamp": "2025-07-29T21:32:19Z",
            "sender": "user_001",
            "receiver": "user_002",
            "text": "Thank you! I'm flattered, especially coming from someone with such an adventurous spirit. It's rare to find someone genuine on here. You seem different. Check out this link: https://example.com/hiking-tips"
        },
        {
            "timestamp": "2025-07-29T21:45:03Z",
            "sender": "user_002",
            "receiver": "user_001",
            "text": "Haha, I try to be. I feel the same way, it's a lot of swiping through the same old stuff. So what do you do when you're not taking amazing photos?"
        },
        {
            "timestamp": "2025-07-29T22:15:30Z",
            "sender": "user_003",
            "receiver": "user_001",
            "text": "Hello gorgeous! You look absolutely stunning in your photos. I have an urgent financial opportunity that could help you. Can you help me transfer some money? I'm in the hospital and need emergency assistance."
        },
        {
            "timestamp": "2025-07-29T22:16:45Z",
            "sender": "user_003",
            "receiver": "user_002",
            "text": "Hey beautiful! Want to see my photos? Click here: http://suspicious-link.com"
        }
    ]
    return sample_data


# Example usage
if __name__ == "__main__":
    # Define some verified users for testing (optional)
    verified_users = {"user_002"}  # Example: user_002 is considered verified
    
    # Option 1: Load from JSON file (if you have one)
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            convo_data = json.load(f)

        print("Loaded conversation data from conversation_data.json")
    except FileNotFoundError:
        # Option 2: Use sample data
        print("No data.json found, using sample data")
        convo_data = load_sample_data()
    
    # Process the conversation data
    user_feats, msg_records = parse_conversation(convo_data, verified_users)
    
    # Save the processed data
    save_processed_data(user_feats, msg_records)
    
    print("\nProcessing complete! Check the 'processed' directory for output files.")