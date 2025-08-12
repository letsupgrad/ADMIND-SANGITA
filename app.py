import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict, Any
import asyncio

# Page configuration
st.set_page_config(
    page_title="AdMind Neural - Predictive AdTech Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for neural theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease-in-out infinite;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .neural-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
        border: 1px solid rgba(255, 107, 107, 0.3);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid #4caf50;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        color: #4caf50;
    }
    
    .trend-up { color: #4caf50; }
    .trend-down { color: #f44336; }
    
    .insight-card {
        background: rgba(15, 15, 15, 0.1);
        border-left: 4px solid #ff6b6b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    
    .rag-response {
        background: rgba(10, 10, 10, 0.05);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'neural_feed' not in st.session_state:
    st.session_state.neural_feed = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = {
        'roas': 4.2,
        'ctr': 2.8,
        'risk': 23,
        'budget_eff': 89
    }
if 'autopilot_active' not in st.session_state:
    st.session_state.autopilot_active = False

class PineconeRAG:
    """RAG implementation with Pinecone vector database"""
    
    def __init__(self, api_key: str, environment: str = "us-east1-gcp"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "admind-knowledge"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self._setup_index()
        
    def _setup_index(self):
        """Initialize Pinecone index"""
        try:
            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            st.error(f"Failed to setup Pinecone index: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to Pinecone index"""
        try:
            vectors = []
            for doc in documents:
                text = f"{doc['title']} {doc['content']}"
                embedding = self.model.encode(text).tolist()
                vectors.append({
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': {
                        'title': doc['title'],
                        'content': doc['content'],
                        'category': doc.get('category', 'general'),
                        'timestamp': doc.get('timestamp', str(datetime.now()))
                    }
                })
            
            if vectors:
                self.index.upsert(vectors)
                return True
        except Exception as e:
            st.error(f"Failed to add documents: {str(e)}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(question).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract relevant information
            context_docs = []
            sources = []
            
            for match in results['matches']:
                if match['score'] > 0.5:  # Relevance threshold
                    context_docs.append(match['metadata']['content'])
                    sources.append({
                        'title': match['metadata']['title'],
                        'score': match['score'],
                        'category': match['metadata'].get('category', 'general')
                    })
            
            # Generate answer using context
            answer = self._generate_answer(question, context_docs)
            
            return {
                'answer': answer,
                'sources': sources,
                'context_count': len(context_docs)
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'context_count': 0
            }
    
    def _generate_answer(self, question: str, context_docs: List[str]) -> str:
        """Generate answer from context using simple template"""
        if not context_docs:
            return "I couldn't find relevant information in the knowledge base to answer your question."
        
        # Simple template-based response (in production, use OpenAI/Anthropic API)
        context = "\n".join(context_docs[:3])  # Use top 3 most relevant docs
        
        if "compare" in question.lower() or "vs" in question.lower():
            return f"Based on the available data, here's a comparison analysis: {context[:500]}... The key differences show varying performance across different metrics and timeframes."
        
        elif "performance" in question.lower():
            return f"Performance analysis from our data shows: {context[:400]}... These metrics indicate strong trends in user engagement and conversion rates."
        
        elif "audience" in question.lower():
            return f"Audience insights reveal: {context[:400]}... These segments show distinct behavioral patterns and preferences."
        
        else:
            return f"Based on our knowledge base: {context[:500]}... This information provides comprehensive insights for your query."

class AdMindDashboard:
    def __init__(self):
        self.rag_system = None
        
    def initialize_rag(self, pinecone_api_key: str):
        """Initialize RAG system with Pinecone"""
        if pinecone_api_key:
            self.rag_system = PineconeRAG(pinecone_api_key)
            self._populate_sample_data()
        
    def _populate_sample_data(self):
        """Add sample AdTech knowledge to Pinecone"""
        sample_docs = [
            {
                'id': 'doc_1',
                'title': 'Q1 2024 Performance Review',
                'content': 'Video ads on social platforms showed 2.1x higher ROAS compared to image ads. Mobile traffic converted 45% better than desktop. iOS users had 340% higher engagement rates.',
                'category': 'performance'
            },
            {
                'id': 'doc_2', 
                'title': 'Creative Best Practices',
                'content': 'Tech enthusiasts respond best to detailed, feature-rich creatives. Premium buyers prefer value proposition over discounts. Mobile-first video content shows 45% higher engagement.',
                'category': 'creative'
            },
            {
                'id': 'doc_3',
                'title': 'Audience Segmentation Analysis',
                'content': 'Tech Enthusiasts: 34% of audience, highest LTV. Mobile Gamers: 28%, peak activity 8-10 PM. Fitness Focused: 22%, weekend conversion +340%. Premium Buyers: 16%, high retention.',
                'category': 'audience'
            },
            {
                'id': 'doc_4',
                'title': 'Budget Optimization Report',
                'content': 'Increasing mobile bid multiplier to 1.35x projected +$12K revenue. Search campaigns performing 156% above benchmark. Social campaigns need optimization.',
                'category': 'budget'
            },
            {
                'id': 'doc_5',
                'title': 'Competitor Intelligence',
                'content': 'Competitors increasing bids on premium keywords. Defensive strategy recommended. Market share analysis shows opportunities in mobile gaming segment.',
                'category': 'competitive'
            }
        ]
        
        if self.rag_system:
            success = self.rag_system.add_documents(sample_docs)
            if success:
                st.success("✅ Knowledge base initialized successfully!")
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">AdMind Neural</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; color: #888; font-size: 1.2rem; margin-bottom: 2rem;">Predictive AdTech Intelligence Platform</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                '<div class="status-indicator">🟢 Neural Engine Active - Processing 847K Data Points</div>',
                unsafe_allow_html=True
            )
    
    def render_predictions(self):
        """Render neural predictions"""
        st.subheader("🧠 Neural Predictions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        predictions = st.session_state.predictions
        
        with col1:
            st.markdown(f"""
            <div class="neural-card">
                <div class="prediction-value">{predictions['roas']:.1f}x</div>
                <div style="color: #888; margin-top: 5px;">Predicted ROAS</div>
                <div class="trend-up" style="margin-top: 8px;">↗ +15%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="neural-card">
                <div class="prediction-value">{predictions['ctr']:.1f}%</div>
                <div style="color: #888; margin-top: 5px;">Expected CTR</div>
                <div class="trend-up" style="margin-top: 8px;">↗ +8%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="neural-card">
                <div class="prediction-value">{predictions['risk']}</div>
                <div style="color: #888; margin-top: 5px;">Risk Score</div>
                <div class="trend-down" style="margin-top: 8px;">↘ -12%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="neural-card">
                <div class="prediction-value">{predictions['budget_eff']}%</div>
                <div style="color: #888; margin-top: 5px;">Budget Efficiency</div>
                <div class="trend-up" style="margin-top: 8px;">↗ +5%</div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_performance_chart(self):
        """Render neural prediction chart"""
        # Generate sample data
        hours = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
        neural_pred = [2.1, 1.8, 3.2, 4.1, 3.8, 4.5, 4.2]
        actual_perf = [2.0, 1.9, 3.1, 3.9, 3.7, 4.3, None]
        
        fig = go.Figure()
        
        # Neural prediction line
        fig.add_trace(go.Scatter(
            x=hours,
            y=neural_pred,
            mode='lines+markers',
            name='Neural Prediction',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8),
            fill='tonexty'
        ))
        
        # Actual performance line
        fig.add_trace(go.Scatter(
            x=hours,
            y=actual_perf,
            mode='lines+markers',
            name='Actual Performance',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Neural Prediction vs Actual Performance",
            xaxis_title="Time",
            yaxis_title="ROAS",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_neural_feed(self):
        """Render neural insights feed"""
        st.subheader("🔮 Neural Feed")
        
        # Add new insight button
        if st.button("Generate New Insight"):
            self.add_neural_insight()
        
        # Display feed
        feed_container = st.container()
        with feed_container:
            if st.session_state.neural_feed:
                for insight in st.session_state.neural_feed[-5:]:  # Show last 5
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>{insight['icon']} {insight['title']}</strong><br>
                        <small style="color: #888;">{insight['content']}</small><br>
                        <small style="color: #666;">{insight['time']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Neural insights will appear here...")
    
    def add_neural_insight(self):
        """Add a new neural insight"""
        insights = [
            {
                'icon': '🤖',
                'title': 'Campaign Anomaly Detected',
                'content': 'iOS traffic showing 340% conversion spike. Auto-scaling budget allocation.',
                'time': 'Just now'
            },
            {
                'icon': '📊',
                'title': 'New Audience Discovered',
                'content': 'Found high-value segment: "Tech professionals, evening browsers" - 5.2x ROAS potential',
                'time': 'Just now'
            },
            {
                'icon': '⚡',
                'title': 'Budget Reallocation Complete',
                'content': 'Moved $5.2K from low-performing segments to high-ROAS audiences.',
                'time': 'Just now'
            },
            {
                'icon': '🎯',
                'title': 'Creative Fatigue Alert',
                'content': 'Ad #B23 performance declining. Preparing replacement variants.',
                'time': 'Just now'
            }
        ]
        
        new_insight = random.choice(insights)
        st.session_state.neural_feed.append(new_insight)
        st.experimental_rerun()
    
    def render_audience_radar(self):
        """Render audience intelligence radar chart"""
        st.subheader("🎯 Audience Neural Map")
        
        # Sample audience data
        categories = ['Tech Enthusiasts', 'Mobile Gamers', 'Fitness Focused', 'Premium Buyers', 'Casual Users']
        values = [85, 72, 68, 91, 45]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Audience Strength',
            line=dict(color='#ff6b6b')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Audience segments
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Primary Segments:**")
            st.markdown("🔴 Tech Enthusiasts (34%)")
            st.markdown("🔵 Mobile Gamers (28%)")
        with col2:
            st.markdown("**Secondary Segments:**")
            st.markdown("🟢 Fitness Focused (22%)")
            st.markdown("🟡 Premium Buyers (16%)")
    
    def render_rag_interface(self):
        """Render RAG query interface"""
        st.subheader("📚 RAG Knowledge Query")
        
        if not self.rag_system:
            st.warning("⚠️ Please configure Pinecone API key in the sidebar to enable RAG functionality.")
            return
        
        # Query input
        query = st.text_area(
            "Ask a question about your campaigns, audience, or market:",
            placeholder="e.g., 'Compare performance of video vs image ads in the last 30 days'",
            height=100
        )
        
        if st.button("🚀 Ask AdMind AI", type="primary"):
            if query.strip():
                with st.spinner("🧠 Processing your query..."):
                    # Simulate processing time
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing RAG query...")
                    progress_bar.progress(25)
                    time.sleep(1)
                    
                    status_text.text("Retrieving from knowledge base...")
                    progress_bar.progress(50)
                    time.sleep(1)
                    
                    status_text.text("Generating augmented response...")
                    progress_bar.progress(75)
                    time.sleep(1)
                    
                    # Get RAG response
                    response = self.rag_system.query(query)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display response
                    st.markdown(f"""
                    <div class="rag-response">
                        <h4>🤖 AdMind AI Response:</h4>
                        <p>{response['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources
                    if response['sources']:
                        st.markdown("**📄 Retrieved Sources:**")
                        cols = st.columns(min(len(response['sources']), 3))
                        for i, source in enumerate(response['sources'][:3]):
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div style="background: rgba(78, 205, 196, 0.1); 
                                           padding: 10px; border-radius: 8px; 
                                           border-left: 3px solid #4ecdc4;">
                                    <small><strong>{source['title']}</strong><br>
                                    Relevance: {source['score']:.2f}<br>
                                    Category: {source['category']}</small>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.error("Please enter a query.")
    
    def render_creative_intelligence(self):
        """Render creative AI interface"""
        st.subheader("🎨 Creative Intelligence")
        
        # Creative prompt input
        creative_prompt = st.text_input(
            "Describe your ideal ad creative:",
            placeholder="e.g., 'Modern fitness app for young professionals'"
        )
        
        if st.button("Generate AI Creative", type="primary"):
            if creative_prompt.strip():
                with st.spinner("🎨 Generating creative concepts..."):
                    time.sleep(2)
                    
                    # Mock creative generation
                    creatives = [
                        {
                            'type': '🖼️ AI Generated Visual',
                            'score': random.randint(80, 95),
                            'description': f'Based on: {creative_prompt[:30]}...',
                            'tag': 'High Performance'
                        },
                        {
                            'type': '📝 Dynamic Copy',
                            'score': random.randint(75, 90),
                            'description': 'Personalized messaging',
                            'tag': 'A/B Test Ready'
                        },
                        {
                            'type': '🎬 Video Concept',
                            'score': random.randint(70, 88),
                            'description': 'Motion graphics template',
                            'tag': 'Social Optimized'
                        }
                    ]
                    
                    cols = st.columns(3)
                    for i, creative in enumerate(creatives):
                        with cols[i]:
                            color = "#4caf50" if creative['score'] > 85 else "#ff9800"
                            st.markdown(f"""
                            <div class="neural-card">
                                <div class="prediction-value" style="color: {color};">{creative['score']}%</div>
                                <div style="margin: 15px 0;">{creative['type']}</div>
                                <small style="color: #888;">{creative['description']}</small><br>
                                <small style="background: rgba(76, 175, 80, 0.2); 
                                              padding: 4px 8px; border-radius: 12px; 
                                              font-size: 11px;">{creative['tag']}</small>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("Please describe your creative concept.")
    
    def render_autopilot_controls(self):
        """Render autopilot control interface"""
        st.subheader("🚀 Auto-Pilot Controls")
        
        # Intelligence level slider
        intelligence_level = st.slider(
            "Intelligence Level",
            min_value=1,
            max_value=100,
            value=75,
            help="Controls how aggressive the AI optimizations will be"
        )
        
        if intelligence_level < 30:
            level_text = f"{intelligence_level}% - Conservative"
        elif intelligence_level < 70:
            level_text = f"{intelligence_level}% - Balanced"
        elif intelligence_level < 90:
            level_text = f"{intelligence_level}% - Aggressive"
        else:
            level_text = f"{intelligence_level}% - Hyper-Aggressive"
        
        st.markdown(f"**Current Setting:** {level_text}")
        
        # Toggle switches
        col1, col2 = st.columns(2)
        with col1:
            auto_bidding = st.checkbox("Auto-Bidding", value=True)
            creative_rotation = st.checkbox("Creative Rotation", value=True)
        with col2:
            audience_expansion = st.checkbox("Audience Expansion", value=False)
            budget_optimization = st.checkbox("Budget Optimization", value=True)
        
        # Autopilot activation
        if st.session_state.autopilot_active:
            if st.button("🔴 Deactivate Neural Autopilot", type="secondary"):
                st.session_state.autopilot_active = False
                st.success("Neural Autopilot deactivated!")
                st.experimental_rerun()
        else:
            if st.button("🚀 Activate Neural Autopilot", type="primary"):
                with st.spinner("🚀 Activating Neural Systems..."):
                    time.sleep(2)
                    st.session_state.autopilot_active = True
                    st.success("✅ Neural Autopilot Active!")
                    # Add autopilot insight
                    insight = {
                        'icon': '🚀',
                        'title': 'Neural Autopilot Activated',
                        'content': 'All optimization systems now running autonomously. Performance monitoring active.',
                        'time': 'Just now'
                    }
                    st.session_state.neural_feed.append(insight)
                    st.experimental_rerun()
    
    def update_predictions(self):
        """Update predictions with small random variations"""
        predictions = st.session_state.predictions
        predictions['roas'] = max(3.0, min(5.0, predictions['roas'] + random.uniform(-0.2, 0.2)))
        predictions['ctr'] = max(1.5, min(4.0, predictions['ctr'] + random.uniform(-0.1, 0.1)))
        predictions['risk'] = max(5, min(50, predictions['risk'] + random.randint(-3, 3)))
        predictions['budget_eff'] = max(70, min(95, predictions['budget_eff'] + random.randint(-2, 2)))

def main():
    """Main application"""
    dashboard = AdMindDashboard()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Pinecone configuration
    st.sidebar.subheader("🔗 Pinecone Setup")
    pinecone_api_key = st.sidebar.text_input(
        "Pinecone API Key",
        type="password",
        help="Enter your Pinecone API key to enable RAG functionality"
    )
    
    if pinecone_api_key:
        dashboard.initialize_rag(pinecone_api_key)
    
    # Auto-update toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Live Updates")
    auto_update = st.sidebar.checkbox("Enable Auto-Updates", value=True)
    
    if auto_update:
        # Update predictions every 30 seconds
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        
        if time.time() - st.session_state.last_update > 30:
            dashboard.update_predictions()
            st.session_state.last_update = time.time()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Data"):
        dashboard.update_predictions()
        st.experimental_rerun()
    
    # Main dashboard
    dashboard.render_header()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Predictions and chart
        dashboard.render_predictions()
        st.markdown("---")
        dashboard.render_performance_chart()
        
        # Optimization suggestions
        st.markdown("### ⚡ Live Optimizations")
        st.success("✓ Increase mobile bid by 15% (High confidence: 94%)")
        st.info("✓ Pause underperforming creative #A47 (Saves $2.3K)")
        st.warning("✓ Expand to 'fitness enthusiasts' segment (+180% ROAS)")
    
    with col2:
        dashboard.render_neural_feed()
        
        # Action buttons
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Execute All", type="primary"):
                with st.spinner("Executing optimizations..."):
                    time.sleep(2)
                    st.success("✅ All optimizations executed!")
        with col_b:
            if st.button("Deep Dive"):
                st.info("🔍 Detailed analysis coming soon...")
    
    # Bottom section
    st.markdown("---")
    
    # Three column layout for bottom panels
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dashboard.render_creative_intelligence()
    
    with col2:
        dashboard.render_audience_radar()
    
    with col3:
        dashboard.render_autopilot_controls()
    
    # Full width RAG section
    st.markdown("---")
    dashboard.render_rag_interface()
    
    # Status footer
    if st.session_state.autopilot_active:
        st.success("🚀 Neural Autopilot is ACTIVE - AI is optimizing your campaigns")
    else:
        st.info("🧠 Neural Engine Ready - Manual control mode")

if __name__ == "__main__":
    main()
