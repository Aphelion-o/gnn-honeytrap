import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import json

#==============================================================================
# GNN Model Definition
#==============================================================================
class HoneytrapGNN(torch.nn.Module):
    """A Heterogeneous Graph Neural Network for detecting honeytrap accounts."""
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        # Heterogeneous convolutions learn from the graph structure
        self.conv1 = HeteroConv({
            ('account', 'sends', 'message'): SAGEConv((-1, -1), hidden_channels),
            ('message', 'received_by', 'account'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('account', 'sends', 'message'): SAGEConv(hidden_channels, hidden_channels),
            ('message', 'received_by', 'account'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')
        
        # Output layer for binary classification on 'account' nodes
        self.lin = torch.nn.Linear(hidden_channels, 2) # 2 classes: Legitimate (0), Honeytrap (1)
    
    def forward(self, x_dict, edge_index_dict):
        # First GNN layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second GNN layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Return classification output for account nodes
        return self.lin(x_dict["account"])

#==============================================================================
# Data Generation and Graph Creation
#==============================================================================

def generate_sample_data(num_users=100, num_messages=500, num_honeytraps=20, ambiguity_level=0.7, noise_std=0.05, emb_dim=768):
    """
    Generates sample data with high ambiguity to challenge the model.
    """
    # 1. Generate User IDs and Honeytrap Designations
    user_ids = [f"user_{i:03d}" for i in range(num_users)]
    honeytrap_users_set = set(sorted(random.sample(user_ids, num_honeytraps)))
    honeytrap_labels = {uid: 1 if uid in honeytrap_users_set else 0 for uid in user_ids}
    honeytrap_users = sorted(list(honeytrap_users_set))

    # 2. Generate Ambiguous User Features
    base_honeytrap_features = {
        "is_verified_interaction": 0.1, "interaction_frequency": 0.9, "bidirectional_ratio": 0.2,
        "conversation_duration_months": 0.1, "message_frequency_stability": 0.2, "topic_diversity": 0.2,
        "suspicious_ratio": 0.8, "link_ratio": 0.7, "sensitive_word_ratio": 0.6, "flirt_or_bait_ratio": 0.9,
    }
    base_legitimate_features = {
        "is_verified_interaction": 0.9, "interaction_frequency": 0.5, "bidirectional_ratio": 0.8,
        "conversation_duration_months": 0.7, "message_frequency_stability": 0.9, "topic_diversity": 0.8,
        "suspicious_ratio": 0.1, "link_ratio": 0.05, "sensitive_word_ratio": 0.05, "flirt_or_bait_ratio": 0.1,
    }

    user_features = {}
    feature_keys = list(base_honeytrap_features.keys())

    for user_id in user_ids:
        is_honeytrap = honeytrap_labels[user_id] == 1
        user_feature_set = {}

        primary_profile = base_honeytrap_features if is_honeytrap else base_legitimate_features
        ambiguity_profile = base_legitimate_features if is_honeytrap else base_honeytrap_features

        for key in feature_keys:
            blended_value = ((1 - ambiguity_level) * primary_profile[key] + ambiguity_level * ambiguity_profile[key])
            noisy_value = blended_value + random.gauss(0, noise_std)
            clipped_value = round(np.clip(noisy_value, 0.0, 1.0), 4)
            user_feature_set[key] = clipped_value
            
        user_features[user_id] = user_feature_set

    # 3. Generate Message Interactions
    messages = []
    for i in range(num_messages):
        sender_id, receiver_id = random.sample(user_ids, 2)
        message = {
            "id": f"msg_{i:03d}",
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message_embeddings": np.random.randn(emb_dim).tolist()
        }
        messages.append(message)
    
    return messages, user_features, honeytrap_labels, honeytrap_users


def create_hetero_graph(messages, user_features, honeytrap_labels=None):
    """Creates a heterogeneous PyG graph from messages and user features."""
    data = HeteroData()
    all_users = sorted(list(user_features.keys()))
    user_to_idx = {user: i for i, user in enumerate(all_users)}
    msg_to_idx = {msg["id"]: i for i, msg in enumerate(messages)}
    
    # Account features
    feature_keys = list(next(iter(user_features.values())).keys())
    account_feature_matrix = []
    for user_id in all_users:
        features = user_features.get(user_id, {})
        account_feature_matrix.append([features.get(key, 0) for key in feature_keys])
    data["account"].x = torch.tensor(account_feature_matrix, dtype=torch.float)
    
    # Message features (embeddings)
    data["message"].x = torch.tensor([msg["message_embeddings"] for msg in messages], dtype=torch.float)
    
    # Edges
    edge_index_sends = [[], []]
    edge_index_receives = [[], []]
    for msg in messages:
        sender_idx = user_to_idx[msg["sender_id"]]
        receiver_idx = user_to_idx[msg["receiver_id"]]
        msg_idx = msg_to_idx[msg["id"]]
        
        edge_index_sends[0].append(sender_idx)
        edge_index_sends[1].append(msg_idx)
        
        edge_index_receives[0].append(msg_idx)
        edge_index_receives[1].append(receiver_idx)
    
    data["account", "sends", "message"].edge_index = torch.tensor(edge_index_sends, dtype=torch.long)
    data["message", "received_by", "account"].edge_index = torch.tensor(edge_index_receives, dtype=torch.long)
    
    # Labels
    if honeytrap_labels:
        data["account"].y = torch.tensor([honeytrap_labels[user_id] for user_id in all_users], dtype=torch.long)
    
    return data, user_to_idx

#==============================================================================
# New Function to Save Data
#==============================================================================
def save_data_to_json(messages, user_features, honeytrap_labels, honeytrap_users):
    """Saves the generated Python data objects to JSON files."""
    print("\n   - Saving generated data to JSON files...")
    with open("messages.json", "w") as f:
        json.dump(messages, f, indent=4)
    with open("user_features.json", "w") as f:
        json.dump(user_features, f, indent=4)
    with open("honeytrap_labels.json", "w") as f:
        json.dump(honeytrap_labels, f, indent=4)
    with open("honeytrap_users.json", "w") as f:
        json.dump(honeytrap_users, f, indent=4)
    print("   - Data saved successfully.")

#==============================================================================
# Training and Evaluation
#==============================================================================

def train_model(model, data, epochs=200, lr=0.005, weight_decay=1e-4):
    """Trains the HoneytrapGNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out, data["account"].y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
    return model, losses

def evaluate_model(model, data, user_to_idx):
    """Evaluates the model and provides a detailed classification report."""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probabilities = F.softmax(out, dim=1)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data["account"].y.cpu().numpy()
        
        accuracy = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="binary")
        
        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(true, pred, target_names=['Legitimate', 'Honeytrap']))
        
        # Detailed predictions for analysis
        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        print("\n--- Detailed Prediction Analysis ---")
        print(f"{'User ID':<15} {'True Label':<12} {'Prediction':<12} {'Confidence':<12} {'Status':<15}")
        print("-" * 75)
        
        for user_idx in range(len(true)):
            user_id = idx_to_user[user_idx]
            true_label = "Honeytrap" if true[user_idx] == 1 else "Legitimate"
            pred_label = "Honeytrap" if pred[user_idx] == 1 else "Legitimate"
            confidence = probabilities[user_idx][pred[user_idx]].item()
            
            if true[user_idx] != pred[user_idx]:
                status = "✗ False Pos" if pred_label == "Honeytrap" else "✗ False Neg"
                print(f"{user_id:<15} {true_label:<12} {pred_label:<12} {confidence:.4f}{'':8} {status:<15}")

        return accuracy, f1

#==============================================================================
# Main Execution
#==============================================================================
def run_pipeline():
    """Runs the full data generation, training, and evaluation pipeline."""
    print("=" * 60)
    print("HONEYTRAP GNN MODEL PIPELINE")
    print("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n1. Generating sample data with high ambiguity...")
    messages, user_features, honeytrap_labels, honeytrap_users = generate_sample_data(
        num_users=1000, num_messages=5000, num_honeytraps=100, ambiguity_level=0.7
    )
    print(f"   - Generated {len(messages)} messages for {len(user_features)} users.")
    print(f"   - {len(honeytrap_users)} users are designated as honeytraps.")
    
    # ** NEW PART: Save the generated data **
    save_data_to_json(messages, user_features, honeytrap_labels, honeytrap_users)

    print("\n2. Creating heterogeneous graph...")
    data, user_to_idx = create_hetero_graph(messages, user_features, honeytrap_labels)
    print("   - Graph created successfully.")
    print(f"   - Account nodes: {data['account'].x.size(0)}, Message nodes: {data['message'].x.size(0)}")

    print("\n3. Initializing and training model...")
    model = HoneytrapGNN(hidden_channels=32)
    trained_model, losses = train_model(model, data, epochs=300, lr=0.003, weight_decay=5e-4)
    
    print("\n4. Evaluating model...")
    accuracy, f1 = evaluate_model(trained_model, data, user_to_idx)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.show()
    
    print("\n" + "=" * 60)
    if f1 > 0.7:
        print("MODEL PERFORMANCE: EXCELLENT on ambiguous data.")
    elif f1 > 0.5:
        print("MODEL PERFORMANCE: GOOD on ambiguous data.")
    else:
        print("MODEL PERFORMANCE: NEEDS IMPROVEMENT on ambiguous data.")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()