"""
AI-Enhanced Streamlit Interface
Transform the solar platform into an AI-powered experience
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List
import json
import pandas as pd
import sys
import os
from pathlib import Path

# Add backend directory to path for imports
backend_path = os.path.join(Path(__file__).parent, 'backend')
sys.path.append(str(backend_path))

# Import all needed agents
from backend.app.agents.input_mapping_agent import InputMappingAgent
from backend.app.agents.location_intelligence_agent import LocationIntelligenceAgent
from backend.app.agents.system_sizing_agent import EnhancedSystemSizingAgent
from backend.app.agents.brand_intelligence_agent import BrandIntelligenceAgent
from backend.app.agents.educational_agent import EducationalAgent
from backend.app.agents.appliance_analysis_agent import ApplianceAnalysisAgent
from backend.app.agents.qa_agent import QAAgent
from backend.app.agents.super_agent import SuperAgent
from backend.app.agents.chat_interface_agent import ChatInterfaceAgent

# Import core components
from backend.app.core.llm_manager import StreamlinedLLMManager
from backend.app.core.tool_manager import ToolManager as StreamlinedToolManager
from backend.app.core.nlp_processor import NLPProcessor

# Import RAG system components
from rag_system.core.rag_engine import RAGEngine
from rag_system.core.document_processor import DocumentProcessor
from rag_system.core.retrieval_system import RetrievalSystem

# Import the AI agents from the other file
try:
    from ai_enhanced_agents import AIEnhancedSolarPlatform, SolarIntelligenceAI
except ImportError:
    # Fallback for environments where the files are treated as one (mocking)
    st.error("Could not import AI agents. Ensure 'ai_enhanced_agents.py' is available.")
    # Define placeholder classes to prevent crash
    class MockLLMService:
        async def llm_analysis(self, prompt: str) -> str: return ""
    class MockAI(MockLLMService):
        async def ai_solar_consultation(self, user_input: str) -> Dict[str, Any]:
            return {
                "ai_consultation": {
                    "ai_analysis": {"location": "Mock Location", "average_kwh_per_month": 500, "system_goals": "Savings"},
                    "ai_predictions": {"irr_prediction_percent": 10.0, "payback_period_years": 6.0},
                    "ai_optimization": {"panel_count": 15, "battery_storage_kwh": 10},
                    "ai_recommendations": "Mock recommendation: Start with a small system."
                },
                "ai_chat_response": "Hello! I am a mock AI expert.",
                "ai_market_insights": "Mock market insights: Prices are stable.",
                "ai_learning_status": "Mock learning complete."
            }
    AIEnhancedSolarPlatform = MockAI

# --- Utility Functions ---

def format_dict_to_markdown(data: Dict[str, Any]) -> str:
    """Formats a dictionary into a clean markdown list for display."""
    md = ""
    for key, value in data.items():
        # Title case and replace underscores
        display_key = key.replace('_', ' ').title().replace('Kwh', 'kWh').replace('Irp', 'IRR')
        
        # Apply formatting
        if isinstance(value, (int, float)):
            if 'percent' in key:
                display_value = f"{value:.2f}%"
            elif 'usd' in key or 'rate' in key:
                display_value = f"${value:.2f}"
            elif 'years' in key or 'count' in key:
                display_value = int(value) if isinstance(value, float) and value.is_integer() else f"{value:.1f}"
            else:
                display_value = f"{value:,}"
        else:
            display_value = str(value)

        md += f"**- {display_key}:** {display_value} \n"
    return md

# --- AI-Enhanced Streamlit App ---
class AISolarPlatform:
    """AI-Powered Solar Intelligence Platform"""
    
    def __init__(self):
        # Initialize AI Platform
        try:
            self.ai_platform = AIEnhancedSolarPlatform()
        except Exception as e:
            st.error(f"Failed to initialize AI Platform: {e}")
            self.ai_platform = None
            
        # Initialize core components
        try:
            self.llm_manager = StreamlinedLLMManager()
        except Exception as e:
            st.error(f"Failed to initialize LLM Manager: {e}")
            self.llm_manager = None
            
        try:
            self.tool_manager = StreamlinedToolManager()
        except Exception as e:
            st.error(f"Failed to initialize Tool Manager: {e}")
            self.tool_manager = None
            
        try:
            self.nlp_processor = NLPProcessor()
        except Exception as e:
            st.error(f"Failed to initialize NLP Processor: {e}")
            self.nlp_processor = None
        
        # Initialize RAG system
        self.rag_engine = RAGEngine()
        self.document_processor = DocumentProcessor()
        self.retrieval_system = RetrievalSystem(self.rag_engine.vector_store)
        
        # Initialize all agents with a resilient constructor helper
        self.agents = {}

        def _safe_init(agent_cls, display_name: str):
            """Try to instantiate agent with injected managers; if that fails due to
            unexpected kwargs, fall back to no-arg construction and inject managers
            on the created instance. Ensure an `agent_name` attribute exists.
            Returns the agent instance or None on failure."""
            try:
                agent = agent_cls(
                    llm_manager=self.llm_manager,
                    tool_manager=self.tool_manager,
                    nlp_processor=self.nlp_processor
                )
            except TypeError as e:
                # Constructor did not accept injected managers, fall back
                try:
                    agent = agent_cls()
                except Exception as e2:
                    st.error(f"Failed to initialize {display_name} Agent: {e2}")
                    return None
                # Inject managers onto the instance if possible
                try:
                    setattr(agent, 'llm_manager', self.llm_manager)
                    setattr(agent, 'tool_manager', self.tool_manager)
                    setattr(agent, 'nlp_processor', self.nlp_processor)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to initialize {display_name} Agent: {e}")
                return None

            # Normalize agent_name for downstream code that expects it
            if not hasattr(agent, 'agent_name'):
                if hasattr(agent, 'name'):
                    agent.agent_name = getattr(agent, 'name')
                else:
                    agent.agent_name = agent.__class__.__name__

            return agent

        # Instantiate agents using the helper (will gracefully handle mismatches)
        self.agents['input_mapping'] = _safe_init(InputMappingAgent, 'Input Mapping')
        self.agents['location'] = _safe_init(LocationIntelligenceAgent, 'Location Intelligence')
        self.agents['sizing'] = _safe_init(EnhancedSystemSizingAgent, 'System Sizing')
        self.agents['brand'] = _safe_init(BrandIntelligenceAgent, 'Brand Intelligence')
        self.agents['appliance'] = _safe_init(ApplianceAnalysisAgent, 'Appliance Analysis')
        self.agents['educational'] = _safe_init(EducationalAgent, 'Educational')
        self.agents['qa'] = _safe_init(QAAgent, 'QA')
        self.agents['chat'] = _safe_init(ChatInterfaceAgent, 'Chat Interface')
        # Initialize super agent for orchestration
        self.agents['super'] = _safe_init(SuperAgent, 'Super')

        # Expose commonly-referenced agent attributes used elsewhere in the app
        self.super_agent = self.agents.get('super')
        
    def render_ai_header(self):
        """AI-powered header with intelligent branding"""
        st.markdown(
            """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
                * { font-family: 'Inter', sans-serif; }
                
                .ai-header {
                    background: linear-gradient(135deg, #4c669f 0%, #3b5998 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                }

                .ai-header h1 {
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                }
                
                .ai-badges {
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    margin-top: 1rem;
                }
                
                .ai-badge {
                    background: rgba(255,255,255,0.2);
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: bold;
                }
                
                .ai-chat-message {
                    background: #f0f4ff;
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                    border-left: 5px solid #4c669f;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                
                .ai-analysis-card, .ai-prediction-card, .ai-optimization-card {
                    padding: 1.5rem;
                    border-radius: 10px;
                    color: white;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }

                .ai-analysis-card {
                    background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                }

                .ai-prediction-card {
                    background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
                }

                .ai-optimization-card {
                    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                }
                
                .stButton>button {
                    background-color: #4c669f;
                    color: white;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                    transition: background-color 0.3s;
                }
                .stButton>button:hover {
                    background-color: #3b5998;
                }
            </style>
            
            <div class="ai-header">
                <h1>Solar AI Intelligence Platform</h1>
                <p>Multi-Modal AI Solar Ecosystem - Where Intelligence Meets Energy</p>
                <div class="ai-badges">
                    <span class="ai-badge">AI-Powered</span>
                    <span class="ai-badge">Intelligent</span>
                    <span class="ai-badge">Predictive</span>
                    <span class="ai-badge">Personalized</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    
    def render_ai_chat_interface(self):
        """AI-powered conversational interface"""
        st.markdown("## AI Solar Expert Consultation")
        st.markdown("**(HINT: Describe your home, monthly bill, and energy goals below)**")
        
        # Initialize chat history
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # --- FIX: Use st.chat_input on the main page (not in a column) ---
        user_input = st.chat_input("Enter your solar requirements, questions, or ideas here:")
        
        # Process input when the user submits
        if user_input:
            
            # Use asyncio to run the async AI consultation function
            async def run_consultation(input_text):
                with st.spinner("AI Agents are working... Analyzing data, running predictions, and optimizing system design..."):
                    # Call the main AI orchestration function
                    results = await self.ai_platform.ai_solar_consultation(input_text)
                    
                    # Store results in session state
                    st.session_state.latest_analysis = results
                    
                    # Add user message to chat history
                    st.session_state.ai_chat_history.append(
                        {"role": "user", "content": input_text}
                    )
                    
                    # Add AI chat response to chat history
                    st.session_state.ai_chat_history.append(
                        {"role": "ai_chat", "content": results['ai_chat_response']}
                    )
            
            # Run the consultation function
            asyncio.run(run_consultation(user_input))
        
        st.markdown("---")
        
        # Display chat history (conversational part)
        st.markdown("### Conversational Timeline")
        for message in st.session_state.ai_chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** *{message['content']}*")
            elif message["role"] == "ai_chat":
                st.markdown(f"<div class='ai-chat-message'>**AI Solar Expert:** {message['content']}</div>", unsafe_allow_html=True)


    def render_ai_results_dashboard(self):
        """Displays the structured AI output in a dashboard format"""
        st.markdown("## Solar System Analysis Dashboard")
        
        # Tabbed interface for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "System Analysis", "Location Intel", "Appliances", 
            "Market Research", "Education"
        ])
        
        with tab1:
            self._render_system_analysis_tab()
            
        with tab2:
            self._render_location_intelligence_tab()
            
        with tab3:
            self._render_appliance_analysis_tab()
            
        with tab4:
            self._render_market_research_tab()
            
        with tab5:
            self._render_educational_content_tab()

    def _render_system_analysis_tab(self):
        """Render system sizing and analysis results"""
        st.subheader("System Sizing & Analysis")
        
        if "latest_analysis" in st.session_state:
            results = st.session_state.latest_analysis
            consultation = results.get("ai_consultation", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='ai-analysis-card'><h3>System Specifications</h3>{format_dict_to_markdown(consultation.get('ai_analysis', {}))}</div>", unsafe_allow_html=True)
            with col2:
                # Get predictions and watt pricing
                predictions = consultation.get('ai_predictions', {})
                watt_pricing = predictions.get('watt_pricing', {})
                
                # If we have watt pricing, add it to the display
                if watt_pricing:
                    pricing_info = {
                        'Cost Per Watt': f"${watt_pricing.get('cost_per_watt', 0):.2f}",
                        'Total System Watts': f"{watt_pricing.get('total_watts', 0):,}",
                        'Total System Cost': f"${watt_pricing.get('total_cost', 0):,.2f}"
                    }
                    predictions.update(pricing_info)
                
                st.markdown(f"<div class='ai-prediction-card'><h3>Financial Analysis</h3>{format_dict_to_markdown(predictions)}</div>", unsafe_allow_html=True)
                
            st.markdown(f"<div class='ai-optimization-card'><h3>Optimized Design</h3>{format_dict_to_markdown(consultation.get('ai_optimization', {}))}</div>", unsafe_allow_html=True)
    
    def _render_location_intelligence_tab(self):
        """Render location-based analysis"""
        st.subheader("Location Intelligence")
        
        # Location input
        location = st.text_input("Enter your location:", key="location_input")
        if st.button("Analyze Location"):
            with st.spinner("Analyzing location data..."):
                if 'location' in self.agents:
                    location_data = asyncio.run(
                        self.agents['location'].analyze_location(location)
                    )
                else:
                    st.error("Location Intelligence Agent not available")
                    location_data = {"error": "Agent not initialized"}
                st.session_state.location_data = location_data
                
        if "location_data" in st.session_state:
            data = st.session_state.location_data
            st.markdown("### Solar Potential Analysis")
            st.write(data)
            
    def _render_appliance_analysis_tab(self):
        """Render appliance analysis interface"""
        st.subheader("Appliance Load Analysis")
        
        # Common appliance selection
        common_appliances = [
            "Select an appliance",
            "Refrigerator",
            "Air Conditioner",
            "Washing Machine",
            "Dishwasher",
            "Television",
            "Computer",
            "Microwave",
            "Water Heater",
            "Other"
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            appliance_choice = st.selectbox(
                "Select Common Appliance:",
                common_appliances,
                key="appliance_select"
            )
            
            if appliance_choice == "Other":
                custom_appliance = st.text_input(
                    "Enter Custom Appliance Name:",
                    key="custom_appliance"
                )
                appliance = custom_appliance
            else:
                appliance = appliance_choice

        with col2:
            # Additional input options
            usage_hours = st.number_input(
                "Daily Usage (hours):",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5,
                key="usage_hours"
            )
            
            quantity = st.number_input(
                "Quantity:",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="quantity"
            )
            
            # Optional power rating input
            has_power_rating = st.checkbox("I know the power rating (Watts)")
            power_rating = None
            if has_power_rating:
                power_rating = st.number_input(
                    "Power Rating (Watts):",
                    min_value=0,
                    max_value=5000,
                    value=100,
                    step=50,
                    key="power_rating"
                )

        if appliance != "Select an appliance" and st.button("Add Appliance"):
            with st.spinner("Analyzing appliance data..."):
                if 'appliance' in self.agents:
                    appliance_data = asyncio.run(
                        self.agents['appliance'].analyze_appliance({
                            "name": appliance,
                            "usage_hours": usage_hours,
                            "quantity": quantity,
                            "power_rating": power_rating
                        })
                    )
                else:
                    st.error("Appliance Analysis Agent not available")
                    appliance_data = {"error": "Agent not initialized"}
                if "appliances" not in st.session_state:
                    st.session_state.appliances = []
                st.session_state.appliances.append(appliance_data)
        
        # Display added appliances in a table
        if "appliances" in st.session_state and st.session_state.appliances:
            st.markdown("### Added Appliances")
            appliance_data = []
            for app in st.session_state.appliances:
                if isinstance(app, dict):
                    appliance_data.append(app)
            
            if appliance_data:
                df = pd.DataFrame(appliance_data)
                st.dataframe(df, use_container_width=True)
            
            # Add total consumption calculation
            if st.button("Calculate Total Consumption"):
                total_consumption = sum(app.get('daily_consumption_kwh', 0) for app in appliance_data)
                st.info(f"Total Daily Consumption: {total_consumption:.2f} kWh")
                
    def _render_market_research_tab(self):
        """Render market research and brand intelligence"""
        st.subheader("Market Research & Brands")
        
        # Brand analysis
        brand = st.text_input("Enter brand name for analysis:", key="brand_input")
        if st.button("Research Brand"):
            with st.spinner("Analyzing brand data..."):
                if 'brand' in self.agents:
                    brand_data = asyncio.run(
                        self.agents['brand'].analyze_brand(brand)
                    )
                else:
                    st.error("Brand Intelligence Agent not available")
                    brand_data = {"error": "Agent not initialized"}
                st.session_state.brand_data = brand_data
                
        if "brand_data" in st.session_state:
            st.markdown("### Brand Analysis")
            st.write(st.session_state.brand_data)
            
    def _render_educational_content_tab(self):
        """Render educational content and resources"""
        st.subheader("Solar Education Center")
        
        # Topic selection
        topic = st.selectbox(
            "Select a topic to learn about:",
            ["Solar Basics", "System Components", "Installation", "Maintenance", "ROI"]
        )
        
        if st.button("Get Information"):
            with st.spinner("Loading educational content..."):
                if 'educational' in self.agents:
                    content = asyncio.run(
                        self.agents['educational'].get_educational_content(topic)
                    )
                else:
                    st.error("Educational Agent not available")
                    content = "Error: Educational content not available."
                st.markdown(content)

    def run(self):
        """Main execution function for the Streamlit App"""
        st.set_page_config(
            page_title="AI Solar Platform",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.render_ai_header()
        
        # Sidebar for system status and tools
        with st.sidebar:
            st.subheader("System Status")
            if st.button("Check System Status"):
                status = self.super_agent.get_agent_status()
                st.json(status)
                
            st.subheader("Tools")
            tool_choice = st.selectbox(
                "Select Tool:",
                ["Document Processing", "RAG Search", "Data Analysis"]
            )
            
            if tool_choice == "Document Processing":
                uploaded_file = st.file_uploader("Upload Document")
                if uploaded_file:
                    doc_result = self.document_processor.process_file(Path(uploaded_file.name))
                    st.write(doc_result)
            
            elif tool_choice == "RAG Search":
                search_query = st.text_input("Search Knowledge Base")
                if search_query:
                    results = self.rag_engine.search(search_query)
                    st.write(results)
            
            elif tool_choice == "Data Analysis":
                data_file = st.file_uploader("Upload Data File")
                if data_file:
                    df = pd.read_csv(data_file)
                    st.write(df.describe())
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main dashboard area
            self.render_ai_results_dashboard()
            
        with col2:
            # Chat interface in sidebar
            self.render_ai_chat_interface()
        
        # Footer
        st.markdown("---")
        st.caption("Powered by AI-Enhanced Solar Platform | © 2023")


if __name__ == "__main__":
    app = AISolarPlatform()
    app.run()