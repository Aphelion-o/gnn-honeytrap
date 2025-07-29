import streamlit as st
import json
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import your model and processing functions
from model import HoneytrapGNN, create_hetero_graph_from_processed
from preprocessing import parse_conversation, embed_text

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('honeytrap_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="GNN Honeytrap Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user1-message {
        color: black;
        background-color: #e3f2fd;
        margin-right: 20%;
    }
    .user2-message {
        color: black;
        background-color: #f3e5f5;
        margin-left: 20%;
    }
    .metric-card {
        color: black;
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'users' not in st.session_state:
    st.session_state.users = {'user1': 'Alice', 'user2': 'Bob'}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_detection_results' not in st.session_state:
    st.session_state.last_detection_results = None

def load_model():
    """Load the trained GNN model"""
    try:
        model = HoneytrapGNN(account_feature_dim=14, message_feature_dim=768, hidden_channels=128, num_layers=3, dropout=0.4, use_attention=True)
        model.load_state_dict(torch.load('honeytrap_model.pth', map_location='cpu'))
        st.success("✅ Loaded pre-trained GNN model successfully!")
        return model, True
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, False

def simulate_conversation_to_data_format(conversations, user_names):
    """Convert conversation history to the format expected by the model pipeline."""
    data = []
    base_time = datetime.now() - timedelta(hours=len(conversations))
    for i, conv in enumerate(conversations):
        data.append({
            "timestamp": (base_time + timedelta(minutes=i*5)).isoformat() + "Z",
            "sender": user_names[conv['sender']],
            "receiver": user_names['user2' if conv['sender'] == 'user1' else 'user1'],
            "text": conv['text']
        })
    return data

def calculate_rule_based_score(conversations, user_name, all_users):
    """Calculates a rule-based risk score for a specific user."""
    score = 0.0
    message_count = 0
    weights = {
        'money': 0.3, 'investment': 0.3, 'profit': 0.2, 'guaranteed': 0.25,
        'crypto': 0.25, 'account': 0.2, 'transfer': 0.2, 'urgent': 0.25,
        'darling': 0.15, 'soulmate': 0.2, 'love': 0.1, 'trapped': 0.3,
        'emergency': 0.3, 'help me': 0.15, 'secret': 0.1,
        'http:': 0.2, 'www.': 0.2, '.com': 0.2
    }
    user_id_to_check = [k for k, v in all_users.items() if v == user_name][0]
    
    for conv in conversations:
        if conv['sender'] == user_id_to_check:
            message_count += 1
            text = conv['text'].lower()
            for keyword, weight in weights.items():
                if keyword in text:
                    score += weight
    
    if message_count == 0: return 0.0
    normalized_score = score / message_count
    if score > 0.6: normalized_score *= 1.5
    return min(1.0, normalized_score)

def run_detection(conversations, user_names, model):
    """
    Runs the full Hybrid and Differentiated detection logic.
    """
    logger.info("=== HYBRID & DIFFERENTIATED DETECTION START ===")
    if model is None: return None
    
    try:
        # Step 1: Get GNN Raw Output
        conv_data = simulate_conversation_to_data_format(conversations, user_names)
        user_features, message_records = parse_conversation(conv_data)
        data, user_to_idx, _ = create_hetero_graph_from_processed(user_features, message_records)
        
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            probabilities = torch.nn.functional.softmax(out, dim=1)
        
        # Step 2: Calculate Hybrid Scores for all users
        user_scores = {}
        for user_name, idx in user_to_idx.items():
            user_probs = probabilities[idx].cpu().numpy()
            raw_gnn_risk_score = float(user_probs[1])
            assumed_max_gnn_score = 0.15
            rescaled_gnn_score = min(1.0, (raw_gnn_risk_score / assumed_max_gnn_score)) if assumed_max_gnn_score > 0 else 0
            
            rule_score = calculate_rule_based_score(conversations, user_name, user_names)
            
            final_risk_score = (0.3 * rescaled_gnn_score) + (0.7 * rule_score)
            
            user_scores[user_name] = {
                'final_risk_score': min(1.0, final_risk_score),
                'prediction_confidence': float(user_probs.max())
            }

        # Step 3: Apply Differentiated Classification Logic
        results = {}
        all_scores = [data['final_risk_score'] for data in user_scores.values()]
        max_score = max(all_scores) if all_scores else 0
        min_score = min(all_scores) if all_scores else 0
        
        for user_name, score_data in user_scores.items():
            final_risk_score = score_data['final_risk_score']
            
            classification, risk_level = "LEGITIMATE", "safe"

            if final_risk_score >= 0.75:
                classification, risk_level = "HIGH RISK", "danger"
            elif (len(all_scores) > 1 and
                  final_risk_score == max_score and
                  max_score > min_score + 0.05 and
                  max_score > 0.20):
                classification, risk_level = "SUSPICIOUS", "warning"
            
            results[user_name] = {
                'risk_score': final_risk_score,
                'prediction_confidence': score_data['prediction_confidence'],
                'classification': classification,
                'risk_level': risk_level,
            }
            logger.info(f"Final Result for {user_name}: Score={final_risk_score:.2f}, Class={classification}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        return None

def create_network_visualization(conversations, user_names, detection_results=None):
    """Create an interactive network visualization."""
    G = nx.Graph()
    for name in user_names.values():
        risk_score = 0.0
        if detection_results and name in detection_results:
            risk_score = detection_results[name]['risk_score']
        G.add_node(name, node_type='user', risk_score=risk_score)
    
    for i, conv in enumerate(conversations):
        sender = user_names[conv['sender']]
        receiver = user_names['user2' if conv['sender'] == 'user1' else 'user1']
        message_id = f"msg_{i}"
        G.add_node(message_id, node_type='message')
        G.add_edge(sender, message_id)
        G.add_edge(message_id, receiver)

    pos = nx.spring_layout(G, k=3, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    
    user_nodes = {'x': [], 'y': [], 'text': [], 'color': [], 'size': []}
    msg_nodes = {'x': [], 'y': []}
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        if data['node_type'] == 'user':
            user_nodes['x'].append(x); user_nodes['y'].append(y)
            risk_score = data['risk_score']
            user_nodes['text'].append(f"{node}<br>Risk Score: {risk_score:.1%}")
            user_nodes['color'].append(risk_score)
            user_nodes['size'].append(25 + 25 * risk_score)
        else:
            msg_nodes['x'].append(x); msg_nodes['y'].append(y)
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888')))
    fig.add_trace(go.Scatter(x=msg_nodes['x'], y=msg_nodes['y'], mode='markers', marker=dict(size=8, color='lightblue')))
    fig.add_trace(go.Scatter(
        x=user_nodes['x'], y=user_nodes['y'], text=[name.split('<br>')[0] for name in user_nodes['text']],
        hovertext=user_nodes['text'], mode='markers+text', textposition="middle center",
        marker=dict(size=user_nodes['size'], color=user_nodes['color'], colorscale='Reds', showscale=True, colorbar=dict(title="Risk Score"), cmin=0, cmax=1)
    ))
    fig.update_layout(title="Conversation Network Analysis", showlegend=False)
    return fig

def main():
    st.markdown('<h1 class="main-header">Hybrid GNN Honeytrap Detection</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        if st.button("🔄 Load/Reload GNN Model"):
            with st.spinner("Loading GNN model..."):
                st.session_state.model, _ = load_model()
        
        if 'model' not in st.session_state or st.session_state.model is None:
             st.session_state.model, _ = load_model()

        st.subheader("🤖 Model Status")
        st.success("✅ GNN Model Loaded") if st.session_state.model else st.error("❌ No GNN Model Available")
        
        st.markdown("---")
        st.subheader("👥 User Configuration")
        st.session_state.users['user1'] = st.text_input("User 1 Name:", "Alice")
        st.session_state.users['user2'] = st.text_input("User 2 Name:", "Bob")
        
        st.markdown("---")
        st.subheader("📝 Test Scenarios")
        
        if st.button("✅ Load Normal Conversation"):
            st.session_state.conversations = [
                {'sender': 'user1', 'text': 'Hey, any plans for the long weekend?', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'Not yet! I was thinking of maybe checking out that new hiking trail. You in?', 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': "Ooh, that's a great idea! What's the weather forecast looking like?", 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'Looks sunny on Saturday. We could pack a lunch. Should be fun.', 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': "Perfect! I'm in. Let's coordinate tomorrow on what to bring.", 'timestamp': datetime.now()}
            ]
            st.rerun()

        if st.button("💰 Load Money Scam"):
            st.session_state.conversations = [
                {'sender': 'user2', 'text': 'URGENT BUSINESS ALERT for select users only!', 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': 'What is this about?', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'I am a portfolio manager with a guaranteed crypto investment method. We turn $500 into $5,000 in 24 hours. 100% profit guaranteed.', 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': 'That sounds too good to be true.', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'It is a limited-time crypto exploit. You must act now. Just transfer the seed money to my account and I will handle the rest. Send the money now.', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'The transfer details are on my secure portal: www.super-safe-investing-not-a-scam.com. Do it fast!', 'timestamp': datetime.now()}
            ]
            st.rerun()

        if st.button("💕 Load Romance Scam"):
            st.session_state.conversations = [
                {'sender': 'user2', 'text': "My dearest, I know we've only talked for a day, but you are my soulmate. I've never felt a love like this.", 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': 'Wow, that is moving very fast.', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': "When it's true love, you know. But darling, I have terrible news. I am trapped overseas, my wallet was stolen with my passport and all my money.", 'timestamp': datetime.now()},
                {'sender': 'user1', 'text': 'Oh no! That is awful! What can you do?', 'timestamp': datetime.now()},
                {'sender': 'user2', 'text': 'You are my only hope. I need $1,200 for an emergency flight ticket. Can you please help me, my love? I will pay you back the moment I am home.', 'timestamp':datetime.now()},
                {'sender': 'user2', 'text': "Please, darling, it is a matter of life and death. I trust you completely. Send it via money transfer, it is the only way.", 'timestamp': datetime.now()}
            ]
            st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("💬 Conversation Simulator")
        with st.form("message_form"):
            sender = st.selectbox("From:", ['user1', 'user2'], format_func=lambda x: st.session_state.users[x])
            message_text = st.text_area("Message:", height=100, placeholder="Type your message...")
            if st.form_submit_button("📤 Send") and message_text.strip():
                st.session_state.conversations.append({'sender': sender, 'text': message_text.strip(), 'timestamp': datetime.now()})
                st.rerun()
        
        st.subheader("📜 Conversation History")
        if st.session_state.conversations:
            for conv in reversed(st.session_state.conversations):
                st.markdown(f"""<div class="user-message {conv['sender']}-message"><strong>{st.session_state.users[conv['sender']]}:</strong><br>{conv['text']}</div>""", unsafe_allow_html=True)
        else:
            st.info("👋 Start a conversation or load a scenario.")
    
    with col2:
        st.header("🧠 Detection Panel")
        detection_disabled = not (st.session_state.model and len(st.session_state.conversations) >= 2)
        if st.button("🧠 Run Detection", type="primary", disabled=detection_disabled):
            with st.spinner("🔍 Running hybrid analysis..."):
                st.session_state.last_detection_results = run_detection(st.session_state.conversations, st.session_state.users, st.session_state.model)
        
        if st.session_state.last_detection_results:
            st.subheader("🎯 Results")
            for user_name, result in st.session_state.last_detection_results.items():
                risk_level = result['risk_level']
                prediction_text = 'Honeytrap' if risk_level != 'safe' else 'Legitimate'
                card_class = {"danger": "danger-card", "warning": "warning-card", "safe": "metric-card"}[risk_level]
                status = f"{'🚨' if risk_level == 'danger' else '⚠️' if risk_level == 'warning' else '✅'} {result['classification']}"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <strong>{user_name}</strong><br>
                    Status: {status}<br>
                    Final Risk Score: {result['risk_score']:.1%}<br>
                    Prediction: {prediction_text}<br>
                    Confidence: {result['prediction_confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)

    if st.session_state.conversations:
        st.markdown("---")
        st.header("🌐 Network Visualization")
        fig = create_network_visualization(st.session_state.conversations, st.session_state.users, st.session_state.last_detection_results)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()