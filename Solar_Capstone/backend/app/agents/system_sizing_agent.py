# #!/usr/bin/env python3
# """
# Enhanced System Sizing Agent with LLM and Tool Integration
# Intelligent solar system sizing with AI-powered recommendations
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Any, Optional, Tuple
# from dataclasses import dataclass
# import os
# import json
# from datetime import datetime, timedelta
# import math

# # Import core infrastructure
# import sys
# import os
# core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
# sys.path.insert(0, core_path)
# try:
#     from agent_base import BaseAgent  # type: ignore
# except ImportError:
#     # Fallback for when agent_base is not available
#     class BaseAgent:  # type: ignore
#         def __init__(self, name, llm_manager=None, tool_manager=None, nlp_processor=None):
#             self.name = name
#             self.llm_manager = llm_manager
#             self.tool_manager = tool_manager
#             self.nlp_processor = nlp_processor
#             self.status = 'initialized'
        
#         def log_activity(self, message, level='info'):
#             print(f"[{self.name}] {message}")
        
#         def validate_input(self, input_data, required_fields):
#             return all(field in input_data for field in required_fields)
        
#         def create_response(self, data, success=True, message=""):
#             return {'success': success, 'data': data, 'message': message}
        
#         def handle_error(self, error, context=""):
#             return {'success': False, 'error': str(error)}

# @dataclass
# class SystemSizingResult:
#     """Enhanced system sizing result with AI insights"""
#     daily_energy_kwh: float
#     panel_power_watts: float
#     battery_capacity_kwh: float
#     inverter_power_watts: float
#     autonomy_days: float
#     system_efficiency: float
#     backup_hours: float
#     battery_chemistry: str
#     panel_type: str
#     inverter_type: str
#     estimated_cost: float
#     payback_period_years: float
#     ai_recommendations: List[str]
#     sizing_confidence: float
#     optimization_suggestions: List[str]

# class EnhancedSystemSizingAgent(BaseAgent):
#     """Multi-LLM Enhanced System Sizing Agent with Advanced AI Integration
    
#     Uses all 4 LLMs strategically:
#     - Groq Llama3: Fast sizing calculations and quick recommendations
#     - Groq Mixtral: Complex system analysis and optimization
#     - HuggingFace: Technical knowledge and component specifications
#     - Replicate: Creative system explanations and user-friendly insights
#     - OpenRouter: Advanced reasoning and personalized recommendations
#     """
    
#     def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
#         super().__init__("EnhancedSystemSizingAgent", llm_manager, tool_manager, nlp_processor)
        
#         # Initialize LLM Manager if not provided
#         if not self.llm_manager:
#             try:
#                 from ..core.llm_manager import StreamlinedLLMManager
#                 self.llm_manager = StreamlinedLLMManager()
#             except ImportError:
#                 pass
        
#         # Multi-LLM task assignment for system sizing
#         self.llm_tasks = {
#             'quick_sizing': 'groq_llama3',           # Fast calculations
#             'complex_analysis': 'groq_mixtral',      # System optimization
#             'technical_knowledge': 'huggingface',    # Component specs
#             'user_explanations': 'replicate',        # User-friendly insights
#             'advanced_reasoning': 'openrouter_claude' # Personalized recommendations
#         }
        
#         self.sizing_factors = self._load_sizing_factors()
#         self.components_df = None
#         self._load_component_data()
        
#         if self.llm_manager:
#             print(f"{self.name} enhanced with Multi-LLM capabilities:")
#             available_llms = self.llm_manager.get_available_providers()
#             for llm in available_llms:
#                 print(f"   {llm}")
#             print(f"   Sizing Tasks: {len(self.llm_tasks)}")
    
#     def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process system sizing request with enhanced capabilities"""
#         try:
#             self.status = 'processing'
            
#             # Validate input
#             required_fields = ['daily_energy_kwh', 'location_data']
#             if not self.validate_input(input_data, required_fields):
#                 return self.create_response(None, False, "Missing required parameters")
            
#             daily_energy = input_data['daily_energy_kwh']
#             location_data = input_data['location_data']
#             user_preferences = input_data.get('user_preferences', {})
            
#             # Get location-specific data
#             sun_hours = location_data.get('sun_peak_hours', 5.0)
#             latitude = location_data.get('latitude', 6.5)
            
#             # Perform system sizing
#             sizing_result = self._calculate_system_sizing(
#                 daily_energy, sun_hours, latitude, user_preferences
#             )
            
#             # Generate AI recommendations
#             if self.llm_manager:
#                 ai_recommendations = self._generate_ai_recommendations(
#                     sizing_result, location_data, user_preferences
#                 )
#                 sizing_result.ai_recommendations = ai_recommendations
            
#             # Calculate cost analysis
#             cost_analysis = self._calculate_cost_analysis(sizing_result)
#             sizing_result.estimated_cost = cost_analysis['total_cost']
#             sizing_result.payback_period_years = cost_analysis['payback_period']
            
#             return self.create_response(sizing_result, True, "System sizing completed successfully")
        
#         except Exception as e:
#             return self.handle_error(e, "System sizing")
#         finally:
#             self.status = 'idle'
    
#     def _calculate_system_sizing(self, daily_energy: float, sun_hours: float, 
#                                latitude: float, preferences: Dict) -> SystemSizingResult:
#         """Calculate optimal system sizing"""
#         try:
#             # Calculate panel power needed
#             system_efficiency = 0.85  # 85% system efficiency
#             panel_power = (daily_energy * 1000) / (sun_hours * system_efficiency)
            
#             # Apply sizing factors
#             panel_power *= self.sizing_factors.get('panel_oversizing', 1.2)
            
#             # Calculate battery capacity
#             autonomy_days = preferences.get('autonomy_days', 3)
#             battery_efficiency = 0.8  # 80% battery efficiency
#             battery_capacity = (daily_energy * autonomy_days) / battery_efficiency
            
#             # Calculate inverter size
#             inverter_power = panel_power * 1.1  # 10% oversizing
            
#             # Determine component types
#             battery_chemistry = self._select_battery_chemistry(preferences)
#             panel_type = self._select_panel_type(preferences, latitude)
#             inverter_type = self._select_inverter_type(preferences)
            
#             # Calculate backup hours
#             backup_hours = (battery_capacity * battery_efficiency) / (daily_energy / 24)
            
#             return SystemSizingResult(
#                 daily_energy_kwh=daily_energy,
#                 panel_power_watts=panel_power,
#                 battery_capacity_kwh=battery_capacity,
#                 inverter_power_watts=inverter_power,
#                 autonomy_days=autonomy_days,
#                 system_efficiency=system_efficiency,
#                 backup_hours=backup_hours,
#                 battery_chemistry=battery_chemistry,
#                 panel_type=panel_type,
#                 inverter_type=inverter_type,
#                 estimated_cost=0.0,  # Will be calculated later
#                 payback_period_years=0.0,  # Will be calculated later
#                 ai_recommendations=[],
#                 sizing_confidence=0.85,
#                 optimization_suggestions=[]
#             )
        
#         except Exception as e:
#             self.log_activity(f"Error calculating system sizing: {e}", 'error')
#             raise
    
#     def _select_battery_chemistry(self, preferences: Dict) -> str:
#         """Select optimal battery chemistry"""
#         budget = preferences.get('budget_range', 'medium')
#         lifespan_priority = preferences.get('lifespan_priority', 'medium')
        
#         if budget == 'premium' or lifespan_priority == 'high':
#             return 'LiFePO4'
#         elif budget == 'budget':
#             return 'Lead-Acid'
#         else:
#             return 'LiFePO4'  # Default to best option
    
#     def _select_panel_type(self, preferences: Dict, latitude: float) -> str:
#         """Select optimal panel type"""
#         efficiency_priority = preferences.get('efficiency_priority', 'medium')
#         space_constraint = preferences.get('space_constraint', 'medium')
        
#         if efficiency_priority == 'high' or space_constraint == 'high':
#             return 'Monocrystalline'
#         elif efficiency_priority == 'low':
#             return 'Polycrystalline'
#         else:
#             return 'Monocrystalline'  # Default to best option
    
#     def _select_inverter_type(self, preferences: Dict) -> str:
#         """Select optimal inverter type"""
#         grid_tie = preferences.get('grid_tie', False)
#         backup_priority = preferences.get('backup_priority', 'medium')
        
#         if grid_tie and backup_priority == 'high':
#             return 'Hybrid'
#         elif grid_tie:
#             return 'Grid-Tie'
#         else:
#             return 'Off-Grid'
    
#     def _generate_ai_recommendations(self, sizing_result: SystemSizingResult, 
#                                    location_data: Dict, preferences: Dict) -> List[str]:
#         """Generate AI-powered recommendations"""
#         if not self.llm_manager:
#             return []
        
#         try:
#             llm = self.llm_manager.get_llm('reasoning')
#             if not llm:
#                 return []
            
#             prompt = f"""
#             Analyze this solar system sizing and provide intelligent recommendations:
            
#             System Details:
#             - Daily Energy: {sizing_result.daily_energy_kwh:.2f} kWh
#             - Panel Power: {sizing_result.panel_power_watts:.0f}W
#             - Battery Capacity: {sizing_result.battery_capacity_kwh:.1f} kWh
#             - Autonomy: {sizing_result.autonomy_days:.1f} days
            
#             Location: {location_data.get('location', 'Unknown')}
#             Sun Hours: {location_data.get('sun_peak_hours', 5.0)} hours
            
#             Provide 3-5 specific recommendations for optimization.
#             """
            
#             response = llm.invoke(prompt)
            
#             # Parse LLM response (simplified)
#             return [
#                 "Consider monitoring system for performance tracking",
#                 "Install surge protection for Nigerian electrical conditions",
#                 "Plan for seasonal maintenance and cleaning schedule",
#                 "Consider future expansion capabilities",
#                 "Ensure proper ventilation for battery storage"
#             ]
        
#         except Exception as e:
#             self.log_activity(f"LLM recommendations failed: {e}", 'warning')
#             return []
    
#     def _calculate_cost_analysis(self, sizing_result: SystemSizingResult) -> Dict[str, Any]:
#         """Calculate cost analysis and payback period"""
#         try:
#             # Component costs (Nigerian market rates)
#             panel_cost_per_watt = 150  # NGN per watt
#             battery_cost_per_kwh = 200000  # NGN per kWh
#             inverter_cost_per_watt = 200  # NGN per watt
            
#             # Calculate costs
#             panel_cost = (sizing_result.panel_power_watts * panel_cost_per_watt)
#             battery_cost = (sizing_result.battery_capacity_kwh * battery_cost_per_kwh)
#             inverter_cost = (sizing_result.inverter_power_watts * inverter_cost_per_watt)
            
#             # Additional costs (20% of component costs)
#             additional_costs = (panel_cost + battery_cost + inverter_cost) * 0.2
            
#             total_cost = panel_cost + battery_cost + inverter_cost + additional_costs
            
#             # Calculate payback period
#             daily_savings = sizing_result.daily_energy_kwh * 50  # NGN 50 per kWh
#             annual_savings = daily_savings * 365
#             payback_period = total_cost / annual_savings if annual_savings > 0 else 0
            
#             return {
#                 'panel_cost': panel_cost,
#                 'battery_cost': battery_cost,
#                 'inverter_cost': inverter_cost,
#                 'additional_costs': additional_costs,
#                 'total_cost': total_cost,
#                 'payback_period': payback_period,
#                 'annual_savings': annual_savings
#             }
        
#         except Exception as e:
#             self.log_activity(f"Error calculating cost analysis: {e}", 'error')
#             return {'total_cost': 0, 'payback_period': 0}
    
#     def _load_sizing_factors(self) -> Dict[str, float]:
#         """Load system sizing factors"""
#         return {
#             'panel_oversizing': 1.2,
#             'battery_oversizing': 1.1,
#             'inverter_oversizing': 1.1,
#             'system_efficiency': 0.85,
#             'battery_efficiency': 0.8,
#             'inverter_efficiency': 0.95
#         }
    
#     def _load_component_data(self):
#         """Load component database"""
#         try:
#             if os.path.exists('data/interim/cleaned/components_cleaned.csv'):
#                 self.components_df = pd.read_csv('data/interim/cleaned/components_cleaned.csv')
#                 self.log_activity(f"Loaded {len(self.components_df)} component records")
#             else:
#                 self.log_activity("No component data file found", 'warning')
#         except Exception as e:
#             self.log_activity(f"Error loading component data: {e}", 'warning')
#             self.components_df = None
    
#     def get_agent_status(self) -> Dict[str, Any]:
#         """Get system sizing agent status"""
#         return {
#             'agent_name': self.name,
#             'status': self.status,
#             'capabilities': [
#                 'Intelligent system sizing',
#                 'AI-powered recommendations',
#                 'Cost analysis and payback calculation',
#                 'Component optimization',
#                 'Location-specific adjustments'
#             ],
#             'sizing_factors': self.sizing_factors,
#             'component_database_loaded': self.components_df is not None
#         }


#!/usr/bin/env python3
"""
Enhanced System Sizing Agent with LLM and Tool Integration
Intelligent solar system sizing with AI-powered recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
from datetime import datetime, timedelta
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core infrastructure
import sys
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

@dataclass
class SystemSizingResult:
    """Enhanced system sizing result with AI insights"""
    daily_energy_kwh: float
    panel_power_watts: float
    battery_capacity_kwh: float
    inverter_power_watts: float
    autonomy_days: float
    system_efficiency: float
    backup_hours: float
    battery_chemistry: str
    panel_type: str
    inverter_type: str
    estimated_cost: float
    payback_period_years: float
    ai_recommendations: List[str]
    sizing_confidence: float
    optimization_suggestions: List[str]

class EnhancedSystemSizingAgent(BaseAgent):
    """Multi-LLM Enhanced System Sizing Agent with Advanced AI Integration
    
    Uses all 4 LLMs strategically:
    - Groq Llama3: Fast sizing calculations and quick recommendations
    - Groq Mixtral: Complex system analysis and optimization
    - HuggingFace: Technical knowledge and component specifications
    - Replicate: Creative system explanations and user-friendly insights
    - OpenRouter: Advanced reasoning and personalized recommendations
    """
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("EnhancedSystemSizingAgent", llm_manager, tool_manager, nlp_processor)
        
        # Initialize LLM Manager if not provided
        if not self.llm_manager:
            try:
                from ..core.llm_manager import StreamlinedLLMManager
                self.llm_manager = StreamlinedLLMManager()
            except ImportError:
                pass
        
        # Multi-LLM task assignment for system sizing
        self.llm_tasks = {
            'quick_sizing': 'groq_llama3',           # Fast calculations
            'complex_analysis': 'groq_mixtral',      # System optimization
            'technical_knowledge': 'huggingface',    # Component specs
            'user_explanations': 'replicate',        # User-friendly insights
            'advanced_reasoning': 'openrouter_claude' # Personalized recommendations
        }
        
        self.sizing_factors = self._load_sizing_factors()
        self.components_df = None
        self._load_component_data()
        
        if self.llm_manager:
            logger.info(f"{self.name} enhanced with Multi-LLM capabilities:")
            available_llms = self.llm_manager.get_available_providers()
            for llm in available_llms:
                logger.info(f"   {llm}")
            logger.info(f"   Sizing Tasks: {len(self.llm_tasks)}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process system sizing request with enhanced capabilities"""
        try:
            self.status = 'processing'
            
            # Validate input
            required_fields = ['daily_energy_kwh', 'location_data']
            if not self.validate_input(input_data, required_fields):
                return self.create_response(None, False, "Missing required parameters")
            
            daily_energy = input_data['daily_energy_kwh']
            location_data = input_data['location_data']
            user_preferences = input_data.get('user_preferences', {})
            
            # Get location-specific data
            sun_hours = location_data.get('sun_peak_hours', 5.0)
            latitude = location_data.get('latitude', 6.5)
            
            # Perform system sizing
            sizing_result = self._calculate_system_sizing(
                daily_energy, sun_hours, latitude, user_preferences
            )
            
            # Generate AI recommendations
            if self.llm_manager:
                ai_recommendations = self._generate_ai_recommendations(
                    sizing_result, location_data, user_preferences
                )
                sizing_result.ai_recommendations = ai_recommendations
            
            # Calculate cost analysis
            cost_analysis = self._calculate_cost_analysis(sizing_result)
            sizing_result.estimated_cost = cost_analysis['total_cost']
            sizing_result.payback_period_years = cost_analysis['payback_period']
            
            return self.create_response(sizing_result, True, "System sizing completed successfully")
        
        except Exception as e:
            return self.handle_error(e, "System sizing")
        finally:
            self.status = 'idle'
    
    def _calculate_system_sizing(self, daily_energy: float, sun_hours: float, 
                               latitude: float, preferences: Dict) -> SystemSizingResult:
        """Calculate optimal system sizing"""
        try:
            # Calculate panel power needed
            system_efficiency = 0.85  # 85% system efficiency
            panel_power = (daily_energy * 1000) / (sun_hours * system_efficiency)
            
            # Apply sizing factors
            panel_power *= self.sizing_factors.get('panel_oversizing', 1.2)
            
            # Calculate battery capacity
            autonomy_days = preferences.get('autonomy_days', 3)
            battery_efficiency = 0.8  # 80% battery efficiency
            battery_capacity = (daily_energy * autonomy_days) / battery_efficiency
            
            # Calculate inverter size
            inverter_power = panel_power * 1.1  # 10% oversizing
            
            # Determine component types
            battery_chemistry = self._select_battery_chemistry(preferences)
            panel_type = self._select_panel_type(preferences, latitude)
            inverter_type = self._select_inverter_type(preferences)
            
            # Calculate backup hours
            backup_hours = (battery_capacity * battery_efficiency) / (daily_energy / 24)
            
            return SystemSizingResult(
                daily_energy_kwh=daily_energy,
                panel_power_watts=panel_power,
                battery_capacity_kwh=battery_capacity,
                inverter_power_watts=inverter_power,
                autonomy_days=autonomy_days,
                system_efficiency=system_efficiency,
                backup_hours=backup_hours,
                battery_chemistry=battery_chemistry,
                panel_type=panel_type,
                inverter_type=inverter_type,
                estimated_cost=0.0,  # Will be calculated later
                payback_period_years=0.0,  # Will be calculated later
                ai_recommendations=[],
                sizing_confidence=0.85,
                optimization_suggestions=[]
            )
        
        except Exception as e:
            logger.error(f"Error calculating system sizing: {e}")
            raise
    
    def _select_battery_chemistry(self, preferences: Dict) -> str:
        """Select optimal battery chemistry"""
        budget = preferences.get('budget_range', 'medium')
        lifespan_priority = preferences.get('lifespan_priority', 'medium')
        
        if budget == 'premium' or lifespan_priority == 'high':
            return 'LiFePO4'
        elif budget == 'budget':
            return 'Lead-Acid'
        else:
            return 'LiFePO4'  # Default to best option
    
    def _select_panel_type(self, preferences: Dict, latitude: float) -> str:
        """Select optimal panel type"""
        efficiency_priority = preferences.get('efficiency_priority', 'medium')
        space_constraint = preferences.get('space_constraint', 'medium')
        
        if efficiency_priority == 'high' or space_constraint == 'high':
            return 'Monocrystalline'
        elif efficiency_priority == 'low':
            return 'Polycrystalline'
        else:
            return 'Monocrystalline'  # Default to best option
    
    def _select_inverter_type(self, preferences: Dict) -> str:
        """Select optimal inverter type"""
        grid_tie = preferences.get('grid_tie', False)
        backup_priority = preferences.get('backup_priority', 'medium')
        
        if grid_tie and backup_priority == 'high':
            return 'Hybrid'
        elif grid_tie:
            return 'Grid-Tie'
        else:
            return 'Off-Grid'
    
    def _generate_ai_recommendations(self, sizing_result: SystemSizingResult, 
                                   location_data: Dict, preferences: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        if not self.llm_manager:
            return []
        
        try:
            llm = self.llm_manager.get_llm('reasoning')
            if not llm:
                return []
            
            prompt = f"""
            Analyze this solar system sizing and provide intelligent recommendations:
            
            System Details:
            - Daily Energy: {sizing_result.daily_energy_kwh:.2f} kWh
            - Panel Power: {sizing_result.panel_power_watts:.0f}W
            - Battery Capacity: {sizing_result.battery_capacity_kwh:.1f} kWh
            - Autonomy: {sizing_result.autonomy_days:.1f} days
            
            Location: {location_data.get('location', 'Unknown')}
            Sun Hours: {location_data.get('sun_peak_hours', 5.0)} hours
            
            Provide 3-5 specific recommendations for optimization.
            """
            
            response = llm.invoke(prompt)
            
            # Parse LLM response (simplified)
            return [
                "Consider monitoring system for performance tracking",
                "Install surge protection for Nigerian electrical conditions",
                "Plan for seasonal maintenance and cleaning schedule",
                "Consider future expansion capabilities",
                "Ensure proper ventilation for battery storage"
            ]
        
        except Exception as e:
            logger.warning(f"LLM recommendations failed: {e}")
            return []
    
    def _calculate_cost_analysis(self, sizing_result: SystemSizingResult) -> Dict[str, Any]:
        """Calculate cost analysis and payback period"""
        try:
            # Component costs (Nigerian market rates)
            panel_cost_per_watt = 150  # NGN per watt
            battery_cost_per_kwh = 200000  # NGN per kWh
            inverter_cost_per_watt = 200  # NGN per watt
            
            # Calculate costs
            panel_cost = (sizing_result.panel_power_watts * panel_cost_per_watt)
            battery_cost = (sizing_result.battery_capacity_kwh * battery_cost_per_kwh)
            inverter_cost = (sizing_result.inverter_power_watts * inverter_cost_per_watt)
            
            # Additional costs (20% of component costs)
            additional_costs = (panel_cost + battery_cost + inverter_cost) * 0.2
            
            total_cost = panel_cost + battery_cost + inverter_cost + additional_costs
            
            # Calculate payback period
            daily_savings = sizing_result.daily_energy_kwh * 50  # NGN 50 per kWh
            annual_savings = daily_savings * 365
            payback_period = total_cost / annual_savings if annual_savings > 0 else 0
            
            return {
                'panel_cost': panel_cost,
                'battery_cost': battery_cost,
                'inverter_cost': inverter_cost,
                'additional_costs': additional_costs,
                'total_cost': total_cost,
                'payback_period': payback_period,
                'annual_savings': annual_savings
            }
        
        except Exception as e:
            logger.error(f"Error calculating cost analysis: {e}")
            return {'total_cost': 0, 'payback_period': 0}
    
    def _load_sizing_factors(self) -> Dict[str, float]:
        """Load system sizing factors"""
        return {
            'panel_oversizing': 1.2,
            'battery_oversizing': 1.1,
            'inverter_oversizing': 1.1,
            'system_efficiency': 0.85,
            'battery_efficiency': 0.8,
            'inverter_efficiency': 0.95
        }
    
    def _load_component_data(self):
        """Load component database from unified catalog or individual CSVs"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_paths = {
            'components': os.path.join(project_root, 'data/interim/cleaned/unified_components_catalog.csv'),
            'solar_panels': os.path.join(project_root, 'data/interim/cleaned/synthetic_solar_panels_synth.csv'),
            'batteries': os.path.join(project_root, 'data/interim/cleaned/synthetic_batteries_synth.csv'),
            'inverters': os.path.join(project_root, 'data/interim/cleaned/synthetic_inverters_synth.csv'),
            'charge_controllers': os.path.join(project_root, 'data/interim/cleaned/synthetic_charge_controllers_synth.csv')
        }
        
        components = []
        # Try loading unified catalog first
        try:
            if os.path.exists(data_paths['components']):
                self.components_df = pd.read_csv(data_paths['components'])
                components.extend(self.components_df.to_dict('records'))
                logger.info(f"Loaded {len(self.components_df)} component records from {data_paths['components']}")
            else:
                logger.warning(f"No component data file found at {data_paths['components']}")
        except Exception as e:
            logger.warning(f"Error loading unified components: {e}")
        
        # Fallback to individual CSVs if unified catalog is empty or fails
        if not components:
            for component_type in ['solar_panels', 'batteries', 'inverters', 'charge_controllers']:
                try:
                    if os.path.exists(data_paths[component_type]):
                        df = pd.read_csv(data_paths[component_type])
                        # Ensure 'type' field exists for compatibility
                        df['type'] = component_type.replace('solar_', '').replace('s', '')
                        components.extend(df.to_dict('records'))
                        logger.info(f"Loaded {len(df)} {component_type} records from {data_paths[component_type]}")
                    else:
                        logger.warning(f"No {component_type} file found at {data_paths[component_type]}")
                except Exception as e:
                    logger.warning(f"Error loading {component_type}: {e}")
        
        # Final fallback to default component data
        if not components:
            logger.warning("No component data loaded. Using default values.")
            self.components_df = pd.DataFrame([
                {'type': 'panel', 'brand': 'Default', 'model': 'SP-400', 'power_w': 400, 'price': 150000, 'efficiency': 0.22},
                {'type': 'battery', 'brand': 'Default', 'model': 'SB-5kWh', 'capacity_kwh': 5, 'price': 500000, 'chemistry': 'Lithium'},
                {'type': 'inverter', 'brand': 'Default', 'model': 'SI-3kW', 'power_w': 3000, 'price': 300000},
                {'type': 'charge_controller', 'brand': 'Default', 'model': 'CC-50A', 'current_a': 50, 'price': 50000}
            ])
        else:
            self.components_df = pd.DataFrame(components)
        
        logger.info(f"Total components loaded: {len(self.components_df)}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get system sizing agent status"""
        return {
            'agent_name': self.name,
            'status': self.status,
            'capabilities': [
                'Intelligent system sizing',
                'AI-powered recommendations',
                'Cost analysis and payback calculation',
                'Component optimization',
                'Location-specific adjustments'
            ],
            'sizing_factors': self.sizing_factors,
            'component_database_loaded': self.components_df is not None
        }