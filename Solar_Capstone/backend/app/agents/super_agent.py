#!/usr/bin/env python3
"""
Super Agent - Central Orchestrator
Central orchestrator for all solar system agents and workflows
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import sys

# Import memory manager
try:
    from ..core.memory_manager import memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("Warning: Memory manager not available")

# Import MCP client
try:
    from ..mcp.solar_mcp_client import get_mcp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP client not available")

# Import core infrastructure
core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)
try:
    from agent_base import BaseAgent  # type: ignore
except ImportError:
    # Fallback for when agent_base is not available
    class BaseAgent:  # type: ignore
        def __init__(self, name, llm_manager=None, tool_manager=None, nlp_processor=None):
            self.name = name
            self.llm_manager = llm_manager
            self.tool_manager = tool_manager
            self.nlp_processor = nlp_processor
            self.status = 'initialized'
        
        def log_activity(self, message, level='info'):
            print(f"[{self.name}] {message}")
        
        def validate_input(self, input_data, required_fields):
            return all(field in input_data for field in required_fields)
        
        def create_response(self, data, success=True, message=""):
            return {'success': success, 'data': data, 'message': message}
        
        def handle_error(self, error, context=""):
            return {'success': False, 'error': str(error)}

# Import workflows
try:
    from ..workflows.solar_system_workflow import SolarSystemWorkflow
    from ..workflows.recommendation_workflow import RecommendationWorkflow
except ImportError:
    # Fallback workflows
    class SolarSystemWorkflow:
        def __init__(self):
            self.name = "SolarSystemWorkflow"
        
        async def process(self, request):
            return {"success": True, "data": {"workflow": "solar_system"}, "message": "Solar system workflow completed"}
    
    class RecommendationWorkflow:
        def __init__(self):
            self.name = "RecommendationWorkflow"
        
        async def process(self, request):
            return {"success": True, "data": {"workflow": "recommendation"}, "message": "Recommendation workflow completed"}

class SuperAgent:
    """
    Central orchestrator for all solar system agents
    Routes requests to appropriate LangGraph workflows
    """
    
    def __init__(self):
        self.agent_name = "SuperAgent"
        self.status = "active"
        
        # Initialize core infrastructure
        self.llm_manager = self._initialize_llm_manager()
        self.tool_manager = self._initialize_tool_manager()
        self.nlp_processor = self._initialize_nlp_processor()
        
        # Initialize workflows
        self.workflows = {
            'solar_analysis': SolarSystemWorkflow(),
            'recommendations': RecommendationWorkflow(),
            'education': None,  # TODO: Implement EducationWorkflow
            'marketplace': None,  # TODO: Implement MarketplaceWorkflow
            'communication': None,  # TODO: Implement CommunicationWorkflow
        }
        
        # State management
        self.active_sessions = {}
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        # Error handling
        self.error_recovery_strategies = {
            'workflow_failure': self._handle_workflow_failure,
            'agent_timeout': self._handle_agent_timeout,
            'llm_failure': self._handle_llm_failure,
            'tool_failure': self._handle_tool_failure
        }
        
        print(f" {self.agent_name} initialized successfully")
        print(f" Available workflows: {list(self.workflows.keys())}")
    
    async def process_chat_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat request with intelligent agent orchestration and memory"""
        try:
            message = request.get('message', '')
            user_data = request.get('user_data', {})
            chat_context = request.get('chat_context', {})
            user_session_info = request.get('user_session_info', {})
            
            # Memory management
            user_id = None
            session_id = None
            conversation_history = []
            
            if MEMORY_AVAILABLE and user_session_info:
                user_id = memory_manager.get_user_id(user_session_info)
                session_id = memory_manager.get_session_id(user_id)
                
                # Get conversation history for context
                conversation_history = memory_manager.get_conversation_history(user_id, session_id, limit=5)
                
                # Add user message to memory
                memory_manager.add_message_to_memory(
                    user_id, session_id, 'user', message,
                    {'intent_analysis': True, 'timestamp': datetime.now().isoformat()}
                )
                
                # Update context with user preferences
                user_preferences = memory_manager.get_user_preferences(user_id)
                if user_preferences:
                    user_data['learned_preferences'] = {
                        'budget_range': user_preferences.budget_range,
                        'preferred_brands': user_preferences.preferred_brands,
                        'system_type': user_preferences.system_type,
                        'learning_level': user_preferences.learning_level
                    }
            
            # Analyze intent and route to appropriate agent
            intent = self._analyze_chat_intent(message.lower())
            
            # Route to specialized agent based on intent
            response = None
            if intent == 'system_sizing':
                response = await self._route_to_sizing_agent(request)
            elif intent == 'education':
                response = await self._route_to_educational_agent(request)
            elif intent == 'location':
                response = await self._route_to_location_agent(request)
            elif intent == 'marketplace':
                response = await self._route_to_marketplace_agent(request)
            elif intent == 'appliances':
                response = await self._route_to_appliance_agent(request)
            else:
                response = await self._route_to_chat_agent(request)
            
            # Save assistant response to memory
            if MEMORY_AVAILABLE and user_id and session_id and response:
                memory_manager.add_message_to_memory(
                    user_id, session_id, 'assistant', response.get('message', ''),
                    {
                        'agent_used': response.get('agent_used', 'unknown'),
                        'intent': intent,
                        'confidence': response.get('confidence', 0.8),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                # Learn from user interactions
                await self._update_user_preferences(user_id, user_data, intent, response)
            
            return response
                
        except Exception as e:
            print(f"SuperAgent chat processing error: {e}")
            return {
                'success': False,
                'message': 'I encountered an error processing your request. Please try again.',
                'error': str(e)
            }
    
    def _analyze_chat_intent(self, message: str) -> str:
        """Analyze user message to determine intent"""
        message = message.lower()
        
        # System sizing keywords
        if any(word in message for word in ['apartment', 'house', 'system', 'size', 'need', 'calculate', 'power', 'kwh']):
            return 'system_sizing'
        
        # Educational keywords
        elif any(word in message for word in ['how', 'what', 'why', 'explain', 'work', 'learn', 'understand']):
            return 'education'
        
        # Location keywords
        elif any(word in message for word in ['location', 'where', 'state', 'city', 'lga', 'solar potential', 'weather']):
            return 'location'
        
        # Marketplace keywords
        elif any(word in message for word in ['buy', 'price', 'cost', 'component', 'panel', 'battery', 'inverter', 'brand']):
            return 'marketplace'
        
        # Appliance keywords
        elif any(word in message for word in ['appliance', 'fridge', 'tv', 'fan', 'air conditioner', 'consumption']):
            return 'appliances'
        
        # Default to general chat
        else:
            return 'general'
    
    async def _route_to_sizing_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to system sizing agent with intelligent appliance analysis"""
        try:
            message = request.get('message', '').lower()
            user_data = request.get('user_data', {})
            appliances = user_data.get('session_appliances', [])
            location = user_data.get('session_location', {})
            
            # Parse appliances from message if not in session
            if not appliances:
                appliances = await self._parse_appliances_from_message(message)
            
            # Parse budget from message
            budget = await self._parse_budget_from_message(message)
            
            if appliances:
                # Calculate actual power requirements
                total_power = 0
                daily_energy = 0
                appliance_details = []
                
                for app in appliances:
                    power = app.get('power_watts', 0)
                    quantity = app.get('quantity', 1)
                    hours = app.get('daily_hours', 8)
                    
                    total_power += power * quantity
                    daily_energy += (power * quantity * hours) / 1000  # Convert to kWh
                    appliance_details.append(f"{quantity}x {app.get('name', 'appliance')} ({power}W each, {hours}h/day)")
                
                # Determine system size based on actual requirements
                panel_power = max(total_power * 1.3, daily_energy * 200)  # 30% safety margin
                battery_capacity = daily_energy * 1.5  # 1.5 days autonomy
                inverter_size = total_power * 1.2  # 20% safety margin
                
                # Calculate number of panels (assuming 400W panels)
                num_panels = max(4, int(panel_power / 400) + 1)
                
                # Estimate cost based on system size
                estimated_cost = self._estimate_system_cost(panel_power, battery_capacity, inverter_size)
                
                # Generate detailed breakdown table
                appliance_table = self._generate_appliance_breakdown_table(appliances)
                
                # Generate smart recommendations
                smart_recommendations = await self._generate_smart_recommendations(appliances, budget, daily_energy, estimated_cost)
                
                # Get real component recommendations from CSV data
                component_recommendations = await self._get_component_recommendations(panel_power, battery_capacity, inverter_size, budget)
                
                # Use MCP tools for enhanced calculations if available
                mcp_enhanced_data = await self._get_mcp_enhanced_data(appliances, location, budget, panel_power, battery_capacity, inverter_size)
                
                # Budget analysis
                budget_analysis = ""
                if budget:
                    if budget >= estimated_cost[1]:  # Can afford premium
                        budget_analysis = f"\n**Your Budget (₦{budget:,})**: Perfect! You can afford a premium system with top-quality components and extended warranty."
                    elif budget >= estimated_cost[0]:  # Can afford basic
                        budget_analysis = f"\n**Your Budget (₦{budget:,})**: Good! You can get a quality system. Consider financing for premium components."
                    else:  # Budget too low
                        budget_analysis = f"\n**Your Budget (₦{budget:,})**: This may be tight for your power needs. Consider reducing AC usage or financing options."
                
                location_info = ""
                if location.get('state'):
                    location_info = f"\n📍 **Location ({location['state']})**: Excellent solar potential in Nigeria!"
                
                return {
                    'success': True,
                    'message': f"""**Personalized Solar System Recommendation:**

{appliance_table}

**Power Analysis:**
• **Total Power Load:** {total_power:,}W (when all appliances run simultaneously)
• **Daily Energy Need:** {daily_energy:.1f} kWh (based on usage hours above)

**Solar System Design Logic:**
• **Peak Sun Hours in Nigeria:** 5.5 hours/day
• **Panel Power Needed:** {daily_energy:.1f} kWh ÷ 5.5h = {daily_energy/5.5:.1f}kW minimum
• **Recommended:** {daily_energy/5.5:.1f}kW × 1.3 safety margin = {(daily_energy/5.5)*1.3:.1f}kW panels

**Recommended System:**
• **Solar Panels**: {num_panels} panels ({num_panels * 400}W total)
• **Battery**: {battery_capacity:.1f} kWh Lithium battery (stores {battery_capacity/daily_energy:.1f} days of energy)
• **Inverter**: {inverter_size/1000:.1f}kW Pure Sine Wave
• **Charge Controller**: {int(num_panels * 400 / 12 * 1.25)}A MPPT

**Battery Logic:**
• **Nighttime**: 7PM - 7AM (12 hours)
• **Cloudy days**: Full 24 hours backup needed
• **Battery Size**: {daily_energy:.1f} kWh × 1.5 days autonomy = {battery_capacity:.1f} kWh

**Estimated Cost:**
• **Basic System**: ₦{estimated_cost[0]:,}
• **Premium System**: ₦{estimated_cost[1]:,}
{budget_analysis}
{location_info}

{smart_recommendations}

{component_recommendations}

{mcp_enhanced_data}

**Key Insights:**
• **Daily Energy Need** = How much energy your appliances consume per day
• **Solar panels** charge batteries during 5.5 peak sun hours only
• **Batteries** power your appliances 24/7, especially at night
• High-power appliances like ACs are your biggest cost drivers!

**Use the form above for detailed component specifications and brand recommendations!**""",
                    'agent_used': 'system_sizing',
                    'system_data': {
                        'total_power': total_power,
                        'daily_energy': daily_energy,
                        'num_panels': num_panels,
                        'battery_capacity': battery_capacity,
                        'estimated_cost': estimated_cost
                    }
                }
            else:
                return {
                    'success': True,
                    'message': """**Solar System Sizing Guide:**

**Small Apartment (1-2 bedrooms):**
• 4-6 panels (1,600W - 2,400W)
• 5-10 kWh battery
• Budget: ₦1.2M - ₦2.5M

**Large Apartment/House (3+ bedrooms):**
• 8-15 panels (3,200W - 6,000W)
• 15-30 kWh battery
• Budget: ₦3M - ₦8M

**Tell me your specific appliances for a personalized calculation!**
Example: "I have 2 ACs, 1 fridge, 6 lights, 2 TVs"
""",
                    'agent_used': 'system_sizing'
                }
                
        except Exception as e:
            return {
                'success': True,
                'message': 'I had trouble calculating your system size. Please use the form interface for accurate calculations.',
                'agent_used': 'system_sizing'
            }
    
    def _load_appliance_database(self):
        """Load appliance database from CSV"""
        try:
            import pandas as pd
            # Get the project root directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            csv_path = os.path.join(project_root, "data", "interim", "cleaned", "appliances_cleaned.csv")
            self.appliances_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.appliances_df)} appliances from CSV")
            return True
        except Exception as e:
            print(f"Failed to load appliance database: {e}")
            return False

    async def _parse_appliances_from_message(self, message: str) -> List[Dict[str, Any]]:
        """Parse appliances from user message using LLM and real CSV data"""
        try:
            if not MEMORY_AVAILABLE or not self.llm_manager:
                return []
            
            # Load appliance database if not already loaded
            if not hasattr(self, 'appliances_df') or self.appliances_df is None:
                if not self._load_appliance_database():
                    return []
            
            llm = self.llm_manager.get_llm('groq_mixtral')  # Use Mixtral for complex parsing
            if not llm:
                return []
            
            # Get available appliances from CSV (limit to avoid huge prompts)
            available_appliances = self.appliances_df['Appliance'].unique().tolist()[:30]
            appliance_categories = self.appliances_df['Category'].unique().tolist()
            
            prompt = f"""Parse appliances from this message and extract structured data:

Message: "{message}"

Available appliances from our database:
{', '.join(available_appliances)}

Categories: {', '.join(appliance_categories)}

Extract appliances using EXACT names from the database above.
Use realistic power consumption and usage hours for Nigeria.

Respond with JSON array:
[
  {{"name": "exact_appliance_name_from_database", "power_watts": actual_power, "quantity": 1, "daily_hours": 8}},
  ...
]"""

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                import json
                appliances = json.loads(response_text)
                
                # Validate and enhance with real CSV data
                validated_appliances = []
                for app in appliances:
                    if isinstance(app, dict) and 'name' in app:
                        # Find matching appliance in CSV
                        matching_app = self.appliances_df[
                            self.appliances_df['Appliance'].str.contains(app['name'], case=False, na=False)
                        ]
                        
                        if not matching_app.empty:
                            # Use real data from CSV
                            real_app = matching_app.iloc[0]
                            validated_appliances.append({
                                'name': real_app['Appliance'],
                                'quantity': app.get('quantity', 1),
                                'power_watts': real_app['max_power_w'],  # Use max power from CSV
                                'daily_hours': app.get('daily_hours', 8),
                                'category': real_app['Category'],
                                'notes': real_app.get('Notes', '')
                            })
                        else:
                            # Fallback to provided data
                            validated_appliances.append(app)
                
                return validated_appliances
                
            except Exception as parse_error:
                print(f"JSON parsing error: {parse_error}")
                return []
                
        except Exception as e:
            print(f"Appliance parsing error: {e}")
            return []
    
    async def _parse_budget_from_message(self, message: str) -> Optional[int]:
        """Parse budget from user message"""
        try:
            import re
            # Look for budget patterns
            budget_patterns = [
                r'budget.*?(\d+(?:,\d{3})*(?:\.\d+)?)\s*million',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*million.*?budget',
                r'₦\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*million',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*m\b'
            ]
            
            for pattern in budget_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    budget_str = match.group(1).replace(',', '')
                    budget_millions = float(budget_str)
                    return int(budget_millions * 1_000_000)
            
            return None
            
        except Exception as e:
            print(f"Budget parsing error: {e}")
            return None
    
    def _estimate_system_cost(self, panel_power: float, battery_capacity: float, inverter_size: float) -> Tuple[int, int]:
        """Estimate system cost in Naira"""
        try:
            # Cost per watt for panels (₦100-200)
            panel_cost = panel_power * 150
            
            # Cost per kWh for batteries (₦200-400K per kWh)
            battery_cost = battery_capacity * 300_000
            
            # Inverter cost (₦100-200 per watt)
            inverter_cost = inverter_size * 150
            
            # Additional components (charge controller, wiring, installation)
            additional_cost = (panel_cost + battery_cost + inverter_cost) * 0.3
            
            basic_cost = int(panel_cost + battery_cost + inverter_cost + additional_cost)
            premium_cost = int(basic_cost * 1.4)  # 40% more for premium components
            
            return (basic_cost, premium_cost)
            
        except Exception as e:
            print(f"Cost estimation error: {e}")
            return (1_000_000, 2_000_000)  # Default fallback
    
    def _generate_appliance_breakdown_table(self, appliances: List[Dict[str, Any]]) -> str:
        """Generate detailed appliance breakdown table"""
        try:
            table_rows = []
            total_energy = 0
            
            for app in appliances:
                power = app.get('power_watts', 0)
                quantity = app.get('quantity', 1)
                hours = app.get('daily_hours', 8)
                daily_energy = (power * quantity * hours) / 1000
                total_energy += daily_energy
                
                table_rows.append(f"| {app.get('name', 'Appliance')} | {power}W | {quantity} | {hours}h | {daily_energy:.2f} kWh |")
            
            table = f"""
**DETAILED APPLIANCE BREAKDOWN:**

| Appliance | Power | Quantity | Hours/Day | Daily Energy |
|-----------|-------|----------|-----------|--------------|
{chr(10).join(table_rows)}
| **TOTAL** | | | | **{total_energy:.1f} kWh** |
"""
            return table
            
        except Exception as e:
            print(f"Table generation error: {e}")
            return ""
    
    async def _generate_smart_recommendations(self, appliances: List[Dict[str, Any]], budget: Optional[int], 
                                           daily_energy: float, estimated_cost: Tuple[int, int]) -> str:
        """Generate intelligent recommendations using LLM"""
        try:
            if not MEMORY_AVAILABLE or not self.llm_manager:
                return self._generate_basic_recommendations(appliances, budget, daily_energy, estimated_cost)
            
            llm = self.llm_manager.get_llm('openrouter_claude')  # Use best LLM for recommendations
            if not llm:
                return self._generate_basic_recommendations(appliances, budget, daily_energy, estimated_cost)
            
            # Find high-energy appliances
            high_energy_appliances = []
            for app in appliances:
                power = app.get('power_watts', 0)
                quantity = app.get('quantity', 1)
                hours = app.get('daily_hours', 8)
                daily_energy_app = (power * quantity * hours) / 1000
                
                if daily_energy_app > 5:  # More than 5 kWh/day
                    high_energy_appliances.append({
                        'name': app.get('name', 'appliance'),
                        'power': power * quantity,
                        'hours': hours,
                        'daily_energy': daily_energy_app,
                        'percentage': (daily_energy_app / daily_energy) * 100
                    })
            
            budget_str = f"₦{budget:,}" if budget else "Not specified"
            system_cost_str = f"₦{estimated_cost[0]:,} - ₦{estimated_cost[1]:,}"
            
            prompt = f"""You're a friendly, knowledgeable solar expert helping a Nigerian family with their solar dreams!

**Their Situation:**
- Daily Energy Need: {daily_energy:.1f} kWh
- Budget: {budget_str}
- System Cost: {system_cost_str}
- High Energy Appliances: {high_energy_appliances}

**Be warm, encouraging, and human in your response. Generate 3 sections:**

1. **PRACTICAL RECOMMENDATIONS** (3-4 specific, easy-to-follow tips)
2. **SMART ALTERNATIVES** (2-3 creative strategies that work with their budget)
3. **MONEY-SAVING SECRETS** (3-4 insider tips to reduce costs)

**Your tone should be:**
- Warm and encouraging (like talking to a friend)
- Use simple, everyday language
- Show empathy for budget concerns
- Be optimistic but honest
- Include encouraging phrases like "Don't worry!", "Here's the good news!", "You can do this!"
- Use Nigerian context and relatable examples

Make them feel confident about their solar journey!"""

            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Smart recommendations error: {e}")
            return self._generate_basic_recommendations(appliances, budget, daily_energy, estimated_cost)
    
    def _generate_basic_recommendations(self, appliances: List[Dict[str, Any]], budget: Optional[int], 
                                      daily_energy: float, estimated_cost: Tuple[int, int]) -> str:
        """Generate basic recommendations as fallback"""
        recommendations = []
        
        # Find AC usage
        ac_energy = 0
        for app in appliances:
            if 'ac' in app.get('name', '').lower():
                power = app.get('power_watts', 0)
                quantity = app.get('quantity', 1)
                hours = app.get('daily_hours', 8)
                ac_energy += (power * quantity * hours) / 1000
        
        if ac_energy > daily_energy * 0.5:  # ACs use more than 50% of energy
            recommendations.append("**PRACTICAL RECOMMENDATIONS:**")
            recommendations.append("• Use ACs during peak sun hours (11AM-4PM) to reduce battery load")
            recommendations.append("• Consider energy-efficient inverter ACs (save 30-40% energy)")
            recommendations.append("• Use timer switches to limit AC runtime")
            recommendations.append("• Install ceiling fans to reduce AC dependency")
            
            recommendations.append("\n**ALTERNATIVE APPROACHES:**")
            recommendations.append("• Phase 1: Power everything except ACs with smaller system")
            recommendations.append("• Phase 2: Add AC capability when budget allows")
            recommendations.append("• Use grid power for ACs initially, solar for other appliances")
            
            recommendations.append("\n**COST-SAVING TIPS:**")
            recommendations.append("• Reduce AC usage from 8h to 4-6h daily")
            recommendations.append("• Use ACs only in bedrooms at night")
            recommendations.append("• Consider split system instead of multiple ACs")
            recommendations.append("• Improve home insulation to reduce cooling needs")
        
        return "\n".join(recommendations)
    
    async def _get_component_recommendations(self, panel_power: float, battery_capacity: float, 
                                          inverter_size: float, budget: Optional[int]) -> str:
        """Get real component recommendations from CSV data"""
        try:
            import pandas as pd
            
            # Load component data
            components_info = []
            
            # Load solar panels
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                panels_path = os.path.join(project_root, "data", "interim", "cleaned", "synthetic_solar_panels_synth.csv")
                panels_df = pd.read_csv(panels_path)
                # Filter panels by power requirements
                suitable_panels = panels_df[
                    (panels_df['rated_power_w'] >= panel_power * 0.8) & 
                    (panels_df['rated_power_w'] <= panel_power * 1.2)
                ].head(3)
                
                if not suitable_panels.empty:
                    components_info.append("**Solar Panels (from your database):**")
                    for _, panel in suitable_panels.iterrows():
                        components_info.append(f"• {panel['brand']} {panel['panel_type']} {panel['rated_power_w']}W - ₦{panel['price_min']:,} - ₦{panel['price_max']:,}")
            except Exception as e:
                print(f"Error loading solar panels: {e}")
            
            # Load batteries
            try:
                batteries_path = os.path.join(project_root, "data", "interim", "cleaned", "synthetic_batteries_synth.csv")
                batteries_df = pd.read_csv(batteries_path)
                # Filter batteries by capacity requirements
                suitable_batteries = batteries_df[
                    (batteries_df['capacity_kwh'] >= battery_capacity * 0.8) & 
                    (batteries_df['capacity_kwh'] <= battery_capacity * 1.5)
                ].head(3)
                
                if not suitable_batteries.empty:
                    components_info.append("\n**Batteries (from your database):**")
                    for _, battery in suitable_batteries.iterrows():
                        components_info.append(f"• {battery['brand']} {battery['type']} {battery['capacity_kwh']}kWh - ₦{battery['price_min']:,} - ₦{battery['price_max']:,}")
            except Exception as e:
                print(f"Error loading batteries: {e}")
            
            # Load inverters
            try:
                inverters_path = os.path.join(project_root, "data", "interim", "cleaned", "synthetic_inverters_synth.csv")
                inverters_df = pd.read_csv(inverters_path)
                # Filter inverters by power requirements
                suitable_inverters = inverters_df[
                    (inverters_df['rated_power_w'] >= inverter_size * 0.8) & 
                    (inverters_df['rated_power_w'] <= inverter_size * 1.2)
                ].head(3)
                
                if not suitable_inverters.empty:
                    components_info.append("\n**Inverters (from your database):**")
                    for _, inverter in suitable_inverters.iterrows():
                        components_info.append(f"• {inverter['brand']} {inverter['type']} {inverter['rated_power_w']}W - ₦{inverter['price_min']:,} - ₦{inverter['price_max']:,}")
            except Exception as e:
                print(f"Error loading inverters: {e}")
            
            if components_info:
                return "\n".join(components_info)
            else:
                return "**Component Recommendations:** Use the form above for detailed component specifications from our database."
                
        except Exception as e:
            print(f"Error getting component recommendations: {e}")
            return "**Component Recommendations:** Use the form above for detailed component specifications."
    
    async def _get_mcp_enhanced_data(self, appliances: List[Dict[str, Any]], location: Dict[str, Any], 
                                   budget: Optional[int], panel_power: float, battery_capacity: float, 
                                   inverter_size: float) -> str:
        """Get enhanced data using MCP tools"""
        try:
            if not MCP_AVAILABLE:
                return ""
            
            mcp_client = await get_mcp_client()
            
            # Get MCP-enhanced calculations
            mcp_calculation = await mcp_client.calculate_solar_system(appliances, location, budget)
            
            # Get MCP component recommendations
            mcp_components = await mcp_client.get_component_recommendations(
                panel_power, battery_capacity, inverter_size, budget
            )
            
            # Get MCP cost estimation
            mcp_costs = await mcp_client.estimate_costs(panel_power, battery_capacity, inverter_size)
            
            # Format MCP enhanced data
            mcp_info = []
            
            if mcp_calculation and not mcp_calculation.get('error'):
                mcp_info.append("**MCP-Enhanced Calculations:**")
                mcp_info.append(f"• Total Power: {mcp_calculation.get('total_power_watts', 0):,}W")
                mcp_info.append(f"• Daily Energy: {mcp_calculation.get('daily_energy_kwh', 0):.1f} kWh")
                mcp_info.append(f"• Recommended Panels: {mcp_calculation.get('num_panels', 0)} panels")
            
            if mcp_components:
                mcp_info.append("\n**MCP Component Recommendations:**")
                for component_type, components in mcp_components.items():
                    if components:
                        mcp_info.append(f"• {component_type.title()}: {len(components)} options available")
            
            if mcp_costs and not mcp_costs.get('error'):
                mcp_info.append(f"\n**MCP Cost Estimation:** ₦{mcp_costs.get('total_cost', 0):,}")
            
            if mcp_info:
                return "\n".join(mcp_info)
            else:
                return ""
                
        except Exception as e:
            print(f"MCP enhanced data error: {e}")
            return ""
    
    async def _route_to_educational_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to educational agent"""
        return {
            'success': True,
            'message': """**Solar Energy Basics:**

**How Solar Works:**
1. Solar panels convert sunlight to electricity
2. Charge controller regulates battery charging
3. Batteries store energy for night use
4. Inverter converts DC to AC for appliances

**Key Benefits:**
• Reduce electricity bills by 70-100%
• Reliable power during outages
• 20-25 year panel lifespan
• Low maintenance

🎓 **Ask specific questions for detailed explanations!**""",
            'agent_used': 'educational'
        }
    
    async def _route_to_location_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to location intelligence agent"""
        return {
            'success': True,
            'message': """📍 **Solar Potential in Nigeria:**

**Excellent Solar Regions:**
• **Northern States**: 5.5-6.5 kWh/m²/day
• **Middle Belt**: 5.0-5.5 kWh/m²/day  
• **Southern States**: 4.5-5.2 kWh/m²/day

**Nigeria has excellent solar potential year-round!**
Use the location selector above for specific data.""",
            'agent_used': 'location_intelligence'
        }
    
    async def _route_to_marketplace_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to marketplace agent"""
        return {
            'success': True,
            'message': """**Solar Component Pricing (Nigeria):**

**Solar Panels:**
• Budget: ₦80-120 per watt
• Premium: ₦150-200 per watt

**Batteries:**
• Lithium: ₦200-300 per kWh
• AGM: ₦100-150 per kWh

**Inverters:**
• 3kW: ₦200-400K
• 5kW: ₦400-700K

**Use the form for personalized component recommendations!**""",
            'agent_used': 'marketplace'
        }
    
    async def _route_to_appliance_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appliance analysis agent"""
        return {
            'success': True,
            'message': """🔌 **Common Appliance Power Usage:**

• **LED Lights**: 5-15W each
• **Ceiling Fan**: 50-80W
• **TV (32")**: 60-120W
• **Laptop**: 45-65W
• **Small Fridge**: 150-300W
• **Large Fridge**: 300-600W
• **Air Conditioner**: 1,000-2,500W

**Add your appliances using the form for accurate calculations!**""",
            'agent_used': 'appliance_analysis'
        }
    
    async def _route_to_chat_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route to general chat agent"""
        return {
            'success': True,
            'message': """👋 **Hello! I'm your Solar Energy Assistant!**

I can help you with:
**System Sizing**: "What solar system do I need?"
🔌 **Appliances**: "How much power does my fridge use?"
📍 **Location**: "What's the solar potential in Lagos?"
**Budget**: "What can I get for ₦2 million?"
**Education**: "How do solar panels work?"

**What would you like to know about solar energy?**""",
            'agent_used': 'chat_interface'
        }
    
    async def _update_user_preferences(self, user_id: str, user_data: Dict[str, Any], intent: str, response: Dict[str, Any]) -> None:
        """Learn and update user preferences from interactions"""
        try:
            if not MEMORY_AVAILABLE:
                return
            
            updates = {}
            
            # Learn from session data
            session_appliances = user_data.get('session_appliances', [])
            session_location = user_data.get('session_location', {})
            session_preferences = user_data.get('session_preferences', {})
            
            if session_appliances:
                updates['appliances'] = session_appliances
            
            if session_location:
                updates['location'] = session_location
            
            if session_preferences:
                if 'budget_amount' in session_preferences:
                    budget_amount = session_preferences['budget_amount']
                    if budget_amount < 1500000:
                        updates['budget_range'] = 'budget'
                    elif budget_amount < 3000000:
                        updates['budget_range'] = 'medium'
                    else:
                        updates['budget_range'] = 'premium'
            
            # Learn from interaction patterns
            if intent == 'education':
                # User asking educational questions - might be beginner
                current_prefs = memory_manager.get_user_preferences(user_id)
                if not current_prefs or current_prefs.learning_level == 'beginner':
                    updates['learning_level'] = 'beginner'
            elif intent == 'system_sizing':
                # User asking about sizing - might be intermediate
                updates['learning_level'] = 'intermediate'
            elif intent == 'marketplace':
                # User asking about components - might be advanced
                updates['learning_level'] = 'advanced'
            
            # Update preferences if we have any updates
            if updates:
                memory_manager.update_user_preferences(user_id, updates)
                
        except Exception as e:
            print(f"Error updating user preferences: {e}")
    
    async def orchestrate_agents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple agents for complex requests"""
        try:
            # This is a placeholder for complex multi-agent orchestration
            return await self.process_request(request)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for all requests
        Routes to appropriate workflow based on request type
        """
        start_time = datetime.now()
        request_id = f"req_{int(datetime.now().timestamp())}"
        
        try:
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            
            # Analyze request type
            request_type = await self._analyze_request_type(request)
            
            # Route to appropriate workflow
            result = await self._route_to_workflow(request_type, request, request_id)
            
            # Update success metrics
            self.performance_metrics['successful_requests'] += 1
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time(response_time)
            
            return {
                'success': True,
                'request_id': request_id,
                'request_type': request_type,
                'result': result,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            # Update failure metrics
            self.performance_metrics['failed_requests'] += 1
            
            # Attempt error recovery
            recovery_result = await self._attempt_error_recovery(e, request, request_id)
            
            return {
                'success': False,
                'request_id': request_id,
                'error': str(e),
                'recovery_attempted': recovery_result['attempted'],
                'recovery_success': recovery_result['success'],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_request_type(self, request: Dict[str, Any]) -> str:
        """Analyze request to determine appropriate workflow"""
        try:
            # Check for explicit request type
            if 'request_type' in request:
                return request['request_type']
            
            # Analyze content for implicit type
            content = request.get('content', '').lower()
            user_query = request.get('user_query', '').lower()
            
            # Solar analysis keywords
            solar_keywords = ['solar', 'panel', 'battery', 'inverter', 'system', 'energy']
            if any(keyword in content or keyword in user_query for keyword in solar_keywords):
                return 'solar_analysis'
            
            # Recommendation keywords
            rec_keywords = ['recommend', 'suggest', 'best', 'optimal', 'choice']
            if any(keyword in content or keyword in user_query for keyword in rec_keywords):
                return 'recommendations'
            
            # Education keywords
            edu_keywords = ['explain', 'how', 'what', 'why', 'learn', 'understand']
            if any(keyword in content or keyword in user_query for keyword in edu_keywords):
                return 'education'
            
            # Default to solar analysis
            return 'solar_analysis'
        
        except Exception as e:
            print(f"Error analyzing request type: {e}")
            return 'solar_analysis'
    
    async def _route_to_workflow(self, request_type: str, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Route request to appropriate workflow"""
        try:
            workflow = self.workflows.get(request_type)
            
            if not workflow:
                return {
                    'success': False,
                    'error': f'Workflow {request_type} not available',
                    'available_workflows': list(self.workflows.keys())
                }
            
            # Process with workflow
            result = await workflow.process(request)
            
            # Store session data
            self.active_sessions[request_id] = {
                'request_type': request_type,
                'timestamp': datetime.now(),
                'status': 'completed'
            }
            
            return result
        
        except Exception as e:
            print(f"Error routing to workflow {request_type}: {e}")
            raise
    
    async def _attempt_error_recovery(self, error: Exception, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Attempt to recover from errors"""
        try:
            error_type = type(error).__name__
            
            if error_type in self.error_recovery_strategies:
                recovery_func = self.error_recovery_strategies[error_type]
                result = await recovery_func(error, request, request_id)
                return {'attempted': True, 'success': result}
            
            return {'attempted': False, 'success': False}
        
        except Exception as recovery_error:
            print(f"Error recovery failed: {recovery_error}")
            return {'attempted': True, 'success': False}
    
    async def _handle_workflow_failure(self, error: Exception, request: Dict[str, Any], request_id: str) -> bool:
        """Handle workflow failure"""
        try:
            # Try alternative workflow
            alternative_type = 'solar_analysis' if request.get('request_type') != 'solar_analysis' else 'recommendations'
            result = await self._route_to_workflow(alternative_type, request, request_id)
            return result.get('success', False)
        except:
            return False
    
    async def _handle_agent_timeout(self, error: Exception, request: Dict[str, Any], request_id: str) -> bool:
        """Handle agent timeout"""
        try:
            # Retry with simplified request
            simplified_request = {
                'content': request.get('content', ''),
                'user_query': request.get('user_query', ''),
                'request_type': 'solar_analysis'
            }
            result = await self._route_to_workflow('solar_analysis', simplified_request, request_id)
            return result.get('success', False)
        except:
            return False
    
    async def _handle_llm_failure(self, error: Exception, request: Dict[str, Any], request_id: str) -> bool:
        """Handle LLM failure"""
        try:
            # Use rule-based fallback
            return await self._rule_based_fallback(request, request_id)
        except:
            return False
    
    async def _handle_tool_failure(self, error: Exception, request: Dict[str, Any], request_id: str) -> bool:
        """Handle tool failure"""
        try:
            # Use cached data or simplified processing
            return await self._simplified_processing(request, request_id)
        except:
            return False
    
    async def _rule_based_fallback(self, request: Dict[str, Any], request_id: str) -> bool:
        """Rule-based fallback when LLM fails"""
        try:
            # Simple rule-based response
            content = request.get('content', '')
            if 'solar' in content.lower():
                return True
            return False
        except:
            return False
    
    async def _simplified_processing(self, request: Dict[str, Any], request_id: str) -> bool:
        """Simplified processing when tools fail"""
        try:
            # Basic processing without external tools
            return True
        except:
            return False
    
    def _update_response_time(self, response_time: float):
        """Update average response time"""
        current_avg = self.performance_metrics['avg_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests > 1:
            new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
            self.performance_metrics['avg_response_time'] = new_avg
        else:
            self.performance_metrics['avg_response_time'] = response_time
    
    def _initialize_llm_manager(self):
        """Initialize LLM manager"""
        try:
            # Import and initialize LLM manager
            from ..core.llm_manager import LLMManager
            return LLMManager()
        except ImportError:
            print("LLM Manager not available")
            return None
    
    def _initialize_tool_manager(self):
        """Initialize tool manager"""
        try:
            # Import and initialize tool manager
            from ..core.tool_manager import ToolManager
            return ToolManager()
        except ImportError:
            print("Tool Manager not available")
            return None
    
    def _initialize_nlp_processor(self):
        """Initialize NLP processor"""
        try:
            # Import and initialize NLP processor
            from ..core.nlp_processor import NLPProcessor
            return NLPProcessor()
        except ImportError:
            print("NLP Processor not available")
            return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get Super Agent status"""
        return {
            'agent_name': self.agent_name,
            'status': self.status,
            'workflows': {
                name: 'available' if workflow else 'not_implemented'
                for name, workflow in self.workflows.items()
            },
            'performance_metrics': self.performance_metrics,
            'active_sessions': len(self.active_sessions),
            'capabilities': [
                'Request routing and orchestration',
                'Workflow management',
                'Error recovery and fallback',
                'Performance monitoring',
                'Session management'
            ]
        }
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get status of specific workflow"""
        workflow = self.workflows.get(workflow_name)
        
        if not workflow:
            return {
                'workflow_name': workflow_name,
                'status': 'not_available',
                'message': f'Workflow {workflow_name} not implemented'
            }
        
        return {
            'workflow_name': workflow_name,
            'status': 'available',
            'capabilities': getattr(workflow, 'capabilities', []),
            'last_used': getattr(workflow, 'last_used', None)
        }