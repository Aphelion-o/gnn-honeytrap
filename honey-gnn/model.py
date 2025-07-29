import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

# --- Enhanced HoneytrapGNN Model Definition ---
class HoneytrapGNN(torch.nn.Module):
    def __init__(self, account_feature_dim=14, message_feature_dim=768, hidden_channels=128, num_layers=3, dropout=0.2, use_attention=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Define convolution type based on attention flag
        if use_attention:
            conv_layer = lambda in_dim, out_dim: GATConv((in_dim, in_dim), out_dim, heads=4, concat=False, dropout=dropout, add_self_loops=False)
        else:
            conv_layer = lambda in_dim, out_dim: SAGEConv((in_dim, in_dim), out_dim)
        
        # Build heterogeneous convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(HeteroConv({
            ('account', 'sends', 'message'): conv_layer(-1, hidden_channels),
            ('message', 'received_by', 'account'): conv_layer(-1, hidden_channels),
        }, aggr='mean'))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(HeteroConv({
                ('account', 'sends', 'message'): conv_layer(hidden_channels, hidden_channels),
                ('message', 'received_by', 'account'): conv_layer(hidden_channels, hidden_channels),
            }, aggr='mean'))
        
        # Feature normalization layers
        self.account_norm = torch.nn.BatchNorm1d(hidden_channels)
        self.message_norm = torch.nn.BatchNorm1d(hidden_channels)
        
        # Classification head with multiple layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 4, 2)
        )
    
    def forward(self, x_dict, edge_index_dict):
        # Apply convolution layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and normalization
            if 'account' in x_dict:
                x_dict['account'] = F.relu(x_dict['account'])
                if x_dict['account'].size(0) > 1:  # Only apply BatchNorm if batch size > 1
                    x_dict['account'] = self.account_norm(x_dict['account'])
                x_dict['account'] = F.dropout(x_dict['account'], p=self.dropout, training=self.training)
            
            if 'message' in x_dict:
                x_dict['message'] = F.relu(x_dict['message'])
                if x_dict['message'].size(0) > 1:  # Only apply BatchNorm if batch size > 1
                    x_dict['message'] = self.message_norm(x_dict['message'])
                x_dict['message'] = F.dropout(x_dict['message'], p=self.dropout, training=self.training)
        
        # Return classification output for account nodes
        return self.classifier(x_dict["account"])

# --- Data Loading and Graph Creation Functions ---
def load_processed_data(processed_dir="processed"):
    """Load the processed data from JSON files"""
    import os
    
    user_features_path = os.path.join(processed_dir, "user_features.json")
    message_records_path = os.path.join(processed_dir, "message_records.json")
    
    with open(user_features_path, 'r') as f:
        user_features = json.load(f)
    
    with open(message_records_path, 'r') as f:
        message_records = json.load(f)
    
    # Add unique IDs to messages if not present
    for i, msg in enumerate(message_records):
        if 'id' not in msg:
            msg['id'] = f"msg_{i:06d}"
    
    return user_features, message_records

def create_hetero_graph_from_processed(user_features, message_records, honeytrap_labels=None):
    """Creates a heterogeneous PyG graph from processed data."""
    data = HeteroData()
    
    # Create user and message mappings
    all_users = set(user_features.keys())
    all_users.update(set(msg["sender_id"] for msg in message_records))
    all_users.update(set(msg["receiver_id"] for msg in message_records))
    
    user_to_idx = {user: i for i, user in enumerate(sorted(all_users))}
    msg_to_idx = {msg["id"]: i for i, msg in enumerate(message_records)}
    
    # Account features - all 14 features from our enhanced processor
    feature_keys = [
        "interaction_frequency", "bidirectional_ratio", "conversation_duration_months",
        "message_frequency_stability", "is_verified_interaction", "link_ratio",
        "sensitive_word_ratio", "flirt_or_bait_ratio", "suspicious_ratio",
        "topic_diversity", "emoji_use_ratio", "media_sent_ratio",
        "avg_message_length", "num_unique_partners"
    ]
    
    account_features = []
    for user_id in sorted(user_to_idx.keys(), key=lambda k: user_to_idx[k]):
        features = user_features.get(user_id, {})
        feature_vector = [features.get(key, 0.0) for key in feature_keys]
        account_features.append(feature_vector)
    
    data["account"].x = torch.tensor(account_features, dtype=torch.float)
    
    # Message features (embeddings)
    message_embeddings = []
    for msg in message_records:
        embedding = msg.get("embedding", [0.0] * 768)  # Default to 768-dim zero vector
        message_embeddings.append(embedding)
    
    data["message"].x = torch.tensor(message_embeddings, dtype=torch.float)
    
    # Create edges
    edge_index_sends = [[], []]
    edge_index_receives = [[], []]
    
    for msg in message_records:
        sender_id = msg["sender_id"]
        receiver_id = msg["receiver_id"]
        msg_id = msg["id"]
        
        if sender_id in user_to_idx and receiver_id in user_to_idx and msg_id in msg_to_idx:
            sender_idx = user_to_idx[sender_id]
            receiver_idx = user_to_idx[receiver_id]
            msg_idx = msg_to_idx[msg_id]
            
            edge_index_sends[0].append(sender_idx)
            edge_index_sends[1].append(msg_idx)
            
            edge_index_receives[0].append(msg_idx)
            edge_index_receives[1].append(receiver_idx)
    
    data["account", "sends", "message"].edge_index = torch.tensor(edge_index_sends, dtype=torch.long)
    data["message", "received_by", "account"].edge_index = torch.tensor(edge_index_receives, dtype=torch.long)
    
    # Labels
    if honeytrap_labels:
        labels = []
        for user_id in sorted(user_to_idx.keys(), key=lambda k: user_to_idx[k]):
            labels.append(honeytrap_labels.get(user_id, 0))
        data["account"].y = torch.tensor(labels, dtype=torch.long)
    
    return data, user_to_idx, msg_to_idx

def generate_enhanced_sample_data(num_users=150, num_messages=800, num_honeytraps=30, ambiguity_level=0.6):
    """
    Generate sample data with realistic honeytrap patterns and all 14 features.
    """
    user_ids = [f"user_{i:03d}" for i in range(num_users)]
    honeytrap_users = random.sample(user_ids, num_honeytraps)
    
    user_features = {}
    for user_id in user_ids:
        is_honeytrap = user_id in honeytrap_users
        
        if is_honeytrap:
            # Honeytrap characteristics: high outreach, low bidirectional, suspicious content
            base_features = {
                "interaction_frequency": np.random.normal(2.5, 0.8),  # High activity
                "bidirectional_ratio": np.random.normal(0.2, 0.1),  # Low responses
                "conversation_duration_months": np.random.normal(1.5, 0.5),  # Short-term
                "message_frequency_stability": np.random.normal(0.8, 0.2),  # Irregular patterns
                "is_verified_interaction": np.random.normal(0.1, 0.05),  # Low verification
                "link_ratio": np.random.normal(0.4, 0.15),  # High link sharing
                "sensitive_word_ratio": np.random.normal(0.3, 0.1),  # Sensitive topics
                "flirt_or_bait_ratio": np.random.normal(0.6, 0.15),  # High flirting
                "suspicious_ratio": np.random.normal(0.7, 0.1),  # High suspicion
                "topic_diversity": np.random.normal(0.3, 0.1),  # Limited topics
                "emoji_use_ratio": np.random.normal(0.7, 0.1),  # High emoji use
                "media_sent_ratio": np.random.normal(0.5, 0.2),  # Moderate media
                "avg_message_length": np.random.normal(80, 20),  # Shorter messages
                "num_unique_partners": np.random.normal(15, 5),  # Many targets
            }
        else:
            # Legitimate user characteristics: balanced, bidirectional, varied content
            base_features = {
                "interaction_frequency": np.random.normal(1.2, 0.5),  # Moderate activity
                "bidirectional_ratio": np.random.normal(0.8, 0.15),  # Good responses
                "conversation_duration_months": np.random.normal(6.0, 2.0),  # Longer relationships
                "message_frequency_stability": np.random.normal(0.4, 0.15),  # More regular
                "is_verified_interaction": np.random.normal(0.7, 0.2),  # Higher verification
                "link_ratio": np.random.normal(0.1, 0.05),  # Low link sharing
                "sensitive_word_ratio": np.random.normal(0.05, 0.03),  # Few sensitive words
                "flirt_or_bait_ratio": np.random.normal(0.2, 0.1),  # Moderate flirting
                "suspicious_ratio": np.random.normal(0.2, 0.1),  # Low suspicion
                "topic_diversity": np.random.normal(0.7, 0.15),  # Varied topics
                "emoji_use_ratio": np.random.normal(0.4, 0.2),  # Moderate emoji use
                "media_sent_ratio": np.random.normal(0.3, 0.15),  # Some media
                "avg_message_length": np.random.normal(120, 30),  # Longer messages
                "num_unique_partners": np.random.normal(5, 2),  # Few partners
            }
        
        # Apply ambiguity and constraints
        features = {}
        for key, base_value in base_features.items():
            # Add ambiguity by blending with opposite class
            if is_honeytrap:
                # Mix legitimate characteristics into honeytrap
                legitimate_value = np.random.normal(0.5, 0.2)  # Generic legitimate value
                mixed_value = (1 - ambiguity_level) * base_value + ambiguity_level * legitimate_value
            else:
                # Mix honeytrap characteristics into legitimate
                honeytrap_value = np.random.normal(0.5, 0.2)  # Generic honeytrap value
                mixed_value = (1 - ambiguity_level) * base_value + ambiguity_level * honeytrap_value
            
            # Apply constraints
            if key in ["interaction_frequency", "conversation_duration_months", "avg_message_length", "num_unique_partners"]:
                features[key] = max(0.1, mixed_value)  # Positive values
            elif key == "message_frequency_stability":
                features[key] = max(0.0, mixed_value)  # Non-negative
            else:  # Ratios between 0 and 1
                features[key] = max(0.0, min(1.0, mixed_value))
        
        user_features[user_id] = features
    
    # Generate messages with embeddings
    messages = []
    for i in range(num_messages):
        sender = random.choice(user_ids)
        receiver = random.choice([u for u in user_ids if u != sender])
        
        # Generate realistic embeddings based on user type
        if sender in honeytrap_users:
            # Honeytrap messages cluster around suspicious topics
            base_embedding = np.random.normal(0.5, 0.3, 768)  # Biased toward suspicious
        else:
            base_embedding = np.random.normal(0.0, 1.0, 768)  # Normal distribution
        
        messages.append({
            "id": f"msg_{i:06d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "embedding": base_embedding.tolist()
        })
    
    honeytrap_labels = {user_id: (1 if user_id in honeytrap_users else 0) for user_id in user_ids}
    
    return user_features, messages, honeytrap_labels, honeytrap_users

# --- Training and Evaluation Functions ---
def train_model_with_validation(model, data, epochs=300, lr=0.001, weight_decay=1e-4, train_ratio=0.8):
    """Train the model with improved but conservative training strategies"""
    
    # Create train/validation split
    num_nodes = data["account"].x.size(0)
    num_train = int(num_nodes * train_ratio)
    
    indices = torch.randperm(num_nodes)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    # Calculate class weights only if severely imbalanced (>3:1 ratio)
    train_labels = data["account"].y[train_mask]
    class_counts = torch.bincount(train_labels)
    imbalance_ratio = class_counts.max().float() / class_counts.min().float()
    
    if imbalance_ratio > 3.0:
        total_samples = len(train_labels)
        class_weights = total_samples / (2.0 * class_counts.float())
        class_weights = torch.clamp(class_weights, min=0.5, max=3.0)  # Limit extreme weights
        print(f"    - Imbalanced data detected (ratio: {imbalance_ratio:.1f}), using class weights: {class_weights.tolist()}")
    else:
        class_weights = None
        print(f"    - Balanced data (ratio: {imbalance_ratio:.1f}), using standard loss")
    
    # Keep original Adam optimizer but with slight improvements
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    
    # More conservative scheduler - only reduce on plateau but with better patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=25, min_lr=lr*0.01
    )
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 50
    
    # Conservative gradient clipping - only if gradients are actually exploding
    max_grad_norm = 5.0  # More lenient than before
    
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        
        # Use weighted loss only if we have class weights
        if class_weights is not None:
            train_loss = F.cross_entropy(out[train_mask], train_labels, weight=class_weights)
        else:
            train_loss = F.cross_entropy(out[train_mask], train_labels)
        
        train_loss.backward()
        
        # Only clip gradients if they're actually large
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if total_norm > max_grad_norm and epoch % 50 == 0:
            print(f"    - Gradient clipping applied at epoch {epoch} (norm: {total_norm:.2f})")
        
        optimizer.step()
        
        # === VALIDATION PHASE ===
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            
            if class_weights is not None:
                val_loss = F.cross_entropy(val_out[val_mask], data["account"].y[val_mask], weight=class_weights)
            else:
                val_loss = F.cross_entropy(val_out[val_mask], data["account"].y[val_mask])
            
            # Calculate accuracies
            train_pred = out[train_mask].argmax(dim=1)
            val_pred = val_out[val_mask].argmax(dim=1)
            
            train_acc = (train_pred == train_labels).float().mean().item()
            val_acc = (val_pred == data["account"].y[val_mask]).float().mean().item()
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Improved model selection - use validation accuracy primarily, loss as tiebreaker
        improved = False
        if val_acc > best_val_acc + 0.005:  # Require meaningful improvement
            improved = True
        elif abs(val_acc - best_val_acc) < 0.005 and val_loss < best_val_loss:  # Similar acc, better loss
            improved = True
        
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        # Early stopping with validation
        if patience_counter >= patience:
            print(f"    - Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
            break
        
        # Progress reporting
        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    - Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")
        
        # Simple overfitting detection - reduce LR if train-val gap is too large
        if epoch > 100 and epoch % 25 == 0:
            recent_train_acc = np.mean(train_accuracies[-10:])
            recent_val_acc = np.mean(val_accuracies[-10:])
            if recent_train_acc > recent_val_acc + 0.15:  # 15% gap indicates overfitting
                print(f"    - Large train-val gap detected ({recent_train_acc-recent_val_acc:.3f}), reducing LR")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"    - Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_out = model(data.x_dict, data.edge_index_dict)
        final_train_acc = (final_out[train_mask].argmax(dim=1) == data["account"].y[train_mask]).float().mean().item()
        final_val_acc = (final_out[val_mask].argmax(dim=1) == data["account"].y[val_mask]).float().mean().item()
    
    print(f"    - Final Training Accuracy: {final_train_acc:.4f}")
    print(f"    - Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"    - Generalization Gap: {final_train_acc - final_val_acc:.4f}")
    
    return model, train_losses, val_losses, train_mask, val_mask

def evaluate_model_comprehensive(model, data, user_to_idx, honeytrap_users, train_mask=None, val_mask=None):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probabilities = F.softmax(out, dim=1)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data["account"].y.cpu().numpy()
        probs_positive = probabilities[:, 1].cpu().numpy()
        
        # Overall metrics
        accuracy = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="binary")
        try:
            auc = roc_auc_score(true, probs_positive)
        except:
            auc = 0.5
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*60}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(true, pred, target_names=['Legitimate', 'Honeytrap']))
        
        # Confusion Matrix
        cm = confusion_matrix(true, pred)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"             Leg  Hon")
        print(f"Actual Leg   {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"       Hon   {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        # Train/Val split performance if available
        if train_mask is not None and val_mask is not None:
            train_acc = accuracy_score(true[train_mask], pred[train_mask])
            val_acc = accuracy_score(true[val_mask], pred[val_mask])
            print(f"\nTrain/Validation Split:")
            print(f"  Training Accuracy:   {train_acc:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Overfitting Gap:     {train_acc - val_acc:.4f}")
        
        # Top suspicious users
        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        user_predictions = []
        
        for user_idx in range(len(true)):
            user_id = idx_to_user[user_idx]
            user_predictions.append({
                'user_id': user_id,
                'true_label': true[user_idx],
                'pred_label': pred[user_idx],
                'honeytrap_prob': probs_positive[user_idx],
                'is_honeytrap': user_id in honeytrap_users
            })
        
        # Sort by honeytrap probability
        user_predictions.sort(key=lambda x: x['honeytrap_prob'], reverse=True)
        
        print(f"\nTop 10 Most Suspicious Users:")
        print(f"{'User ID':<12} {'True':<8} {'Pred':<8} {'Prob':<8} {'Status':<12}")
        print("-" * 55)
        
        for i, up in enumerate(user_predictions[:10]):
            true_label = "Hon" if up['true_label'] == 1 else "Leg"
            pred_label = "Hon" if up['pred_label'] == 1 else "Leg"
            status = "✓" if up['true_label'] == up['pred_label'] else "✗"
            print(f"{up['user_id']:<12} {true_label:<8} {pred_label:<8} {up['honeytrap_prob']:.3f}{'':4} {status:<12}")
        
        return accuracy, f1, auc, probabilities.cpu().numpy()

def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Smooth the curves for better visualization
    window = min(10, len(train_losses) // 10)
    if window > 1:
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        plt.plot(train_smooth, label='Training (Smoothed)', alpha=0.8)
        plt.plot(val_smooth, label='Validation (Smoothed)', alpha=0.8)
    plt.title('Smoothed Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

# --- Main Test Execution ---
def run_comprehensive_test():
    print("=" * 70)
    print("ENHANCED HONEYTRAP GNN MODEL TEST (ALL 14 FEATURES)")
    print("=" * 70)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Try to load processed data, otherwise generate sample data
    try:
        print("\n1. Loading processed data...")
        user_features, message_records = load_processed_data()
        
        # For testing, create some honeytrap labels
        user_ids = list(user_features.keys())
        num_honeytraps = max(5, len(user_ids) // 10)  # 10% honeytraps
        # honeytrap_users = random.sample(user_ids, num_honeytraps)
        honeytrap_users = [
    "user_001",
    "user_301",
    "user_401",
    "user_501",
    "user_601",
    "user_701",
    "user_801",
    "user_901",
    "user_1001",
    "user_1101",
    "user_1301",
    "user_1401",
    "user_2402",
    "user_2502",
    "user_3001",
    "user_3102"
]
        honeytrap_labels = {user_id: (1 if user_id in honeytrap_users else 0) for user_id in user_ids}
        
        print(f"    - Loaded {len(message_records)} messages")
        print(f"    - {len(user_features)} users total")
        print(f"    - Simulated {len(honeytrap_users)} honeytrap users")
        
    except FileNotFoundError:
        print("\n1. No processed data found, generating enhanced sample data...")
        user_features, message_records, honeytrap_labels, honeytrap_users = generate_enhanced_sample_data(
            num_users=200, num_messages=1000, num_honeytraps=40, ambiguity_level=0.5
        )
        
        print(f"    - Generated {len(message_records)} messages")
        print(f"    - {len(user_features)} users total")
        print(f"    - {len(honeytrap_users)} honeytrap users ({len(honeytrap_users)/len(user_features)*100:.1f}%)")
    
    # Feature analysis
    print(f"\n2. Feature Analysis:")
    honeytrap_features = [user_features[user] for user in honeytrap_users if user in user_features]
    legitimate_features = [user_features[user] for user in user_features.keys() if user not in honeytrap_users]
    
    key_features = ["suspicious_ratio", "flirt_or_bait_ratio", "bidirectional_ratio", "link_ratio"]
    for feature in key_features:
        if honeytrap_features and legitimate_features:
            hon_avg = np.mean([f[feature] for f in honeytrap_features])
            leg_avg = np.mean([f[feature] for f in legitimate_features])
            print(f"    - {feature}: Honeytrap={hon_avg:.3f}, Legitimate={leg_avg:.3f}")
    
    print(f"\n3. Creating heterogeneous graph...")
    data, user_to_idx, msg_to_idx = create_hetero_graph_from_processed(user_features, message_records, honeytrap_labels)
    
    print(f"    - Account nodes: {data['account'].x.size(0)}")
    print(f"    - Message nodes: {data['message'].x.size(0)}")
    print(f"    - Account features: {data['account'].x.size(1)} (all 14 features)")
    print(f"    - Message features: {data['message'].x.size(1)}")
    print(f"    - Sends edges: {data['account', 'sends', 'message'].edge_index.size(1)}")
    print(f"    - Receives edges: {data['message', 'received_by', 'account'].edge_index.size(1)}")
    
    print(f"\n4. Initializing enhanced model...")
    model = HoneytrapGNN(
        account_feature_dim=14,  # All our features
        message_feature_dim=768,  # MobileBERT embeddings
        hidden_channels=128,
        num_layers=3,
        dropout=0.3,
        use_attention=True
    )
    
    # Initialize model
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    - Total parameters: {total_params:,}")
    
    print(f"\n5. Training model with validation...")
    model, train_losses, val_losses, train_mask, val_mask = train_model_with_validation(
        model, data, epochs=400, lr=0.001, weight_decay=1e-4
    )
    
    print(f"\n6. Comprehensive evaluation...")
    accuracy, f1, auc, probabilities = evaluate_model_comprehensive(
        model, data, user_to_idx, honeytrap_users, train_mask, val_mask
    )
    
    print(f"\n7. Baseline comparison...")
    majority_class_count = max(len(honeytrap_users), len(user_features) - len(honeytrap_users))
    baseline_accuracy = majority_class_count / len(user_features)
    print(f"    - Majority class baseline: {baseline_accuracy:.3f}")
    print(f"    - Model accuracy: {accuracy:.3f}")
    print(f"    - Improvement over baseline: {accuracy - baseline_accuracy:.3f}")
    
    # Plot training curves
    try:
        plot_training_curves(train_losses, val_losses, 'enhanced_training_curves.png')
    except Exception as e:
        print(f"    - Error saving plots: {e}")
    
    print(f"\n" + "=" * 70)
    if accuracy > 0.85 and f1 > 0.8 and auc > 0.9:
        print("MODEL PERFORMANCE: EXCELLENT ⭐⭐⭐")
    elif accuracy > 0.75 and f1 > 0.7 and auc > 0.8:
        print("MODEL PERFORMANCE: GOOD ⭐⭐")
    elif accuracy > 0.65 and f1 > 0.6 and auc > 0.7:
        print("MODEL PERFORMANCE: FAIR ⭐")
    else:
        print("MODEL PERFORMANCE: NEEDS IMPROVEMENT")
    print("=" * 70)
    
    return model, data, user_to_idx, honeytrap_users

def save_trained_model():
    """Train a model and save it for the frontend"""
    print("Training model for frontend use...")

    # Run the comprehensive test which trains the model
    model, data, user_mapping, honeytrap_list = run_comprehensive_test() # Call the function directly

    # Save the trained model
    torch.save(model.state_dict(), 'honeytrap_model.pth')
    print("\n:white_check_mark: Model saved as 'honeytrap_model.pth'")
    print("You can now run the Streamlit frontend!")

    return model


if __name__ == "__main__":
    model, data, user_mapping, honeytrap_list = run_comprehensive_test()
    save_trained_model()