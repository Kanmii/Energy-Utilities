"""
Appliance Analysis Agent with NLP + LLM Integration
Intelligent appliance mapping and energy calculation with natural language understanding
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import logging
from datetime import datetime
try:
    from fuzzywuzzy import fuzz, process  # type: ignore
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not available, using basic string matching")

# Import core infrastructure
import sys
import os
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
class ApplianceAnalysis:
    """Enhanced appliance analysis with AI insights"""
    appliance_name: str
    matched_appliance: str
    power_watts: float
    usage_hours: float
    quantity: int
    daily_energy_kwh: float
    confidence_score: float
    ai_explanation: str  # LLM-generated explanation
    usage_pattern: str  # AI-identified usage pattern
    energy_tips: List[str]  # AI-generated energy saving tips
    alternatives: List[str]  # AI-suggested alternatives

class ApplianceAnalysisAgent(BaseAgent):
    """Appliance analysis agent with NLP and LLM capabilities"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("ApplianceAnalysisAgent", llm_manager, tool_manager, nlp_processor)
        self.appliances_df = None
        self.synonyms_db = {}
        self._load_appliance_data()
        self._load_synonyms()
    
    async def analyze_appliance(self, appliance_text: str) -> Dict[str, Any]:
        """Analyze a given appliance description"""
        input_data = {
            'appliances': [{
                'name': appliance_text,
                'usage_hours': 8,  # Default usage hours
                'quantity': 1      # Default quantity
            }],
            'user_query': f"Analyze energy usage for {appliance_text}"
        }
        return self.process(input_data)
 
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process appliance analysis request with AI intelligence"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['appliances']):
                return self.create_response(None, False, "Missing appliance information")
            
            appliances = input_data['appliances']
            user_query = input_data.get('user_query', '')
            
            # NLP Analysis: Understand appliance descriptions
            if user_query and self.nlp_processor:
                nlp_analysis = self.nlp_processor.process_input(user_query)
                enhanced_appliances = self._enhance_appliances_with_nlp(appliances, nlp_analysis)
            else:
                enhanced_appliances = appliances
            
            # Process appliances with AI intelligence
            appliance_analyses = []
            total_energy = 0
            
            for appliance in enhanced_appliances:
                analysis = self._analyze_appliance_with_ai(appliance, user_query)
                appliance_analyses.append(analysis)
                total_energy += analysis.daily_energy_kwh
            
            # Generate system insights
            system_insights = self._generate_system_insights(appliance_analyses, total_energy)
            
            return self.create_response({
                'appliance_analyses': appliance_analyses,
                'total_daily_energy_kwh': total_energy,
                'system_insights': system_insights
            }, True, "AI-powered appliance analysis completed")
            
        except Exception as e:
            return self.handle_error(e, "Appliance analysis")
        finally:
            self.status = 'idle'
 
    def _enhance_appliances_with_nlp(self, appliances: List[Dict], nlp_analysis: Dict) -> List[Dict]:
        """Enhance appliance understanding using NLP analysis"""
        enhanced_appliances = []
        
        for appliance in appliances:
            enhanced = appliance.copy()
            
            # Extract usage patterns from NLP
            if 'keywords' in nlp_analysis:
                keywords = nlp_analysis['keywords']
                
                # Extract usage intensity
                if any(word in keywords for word in ['heavy', 'intensive', 'constant']):
                    enhanced['usage_intensity'] = 'high'
                elif any(word in keywords for word in ['light', 'occasional', 'rare']):
                    enhanced['usage_intensity'] = 'low'
                else:
                    enhanced['usage_intensity'] = 'medium'
                
                # Extract usage timing
                if any(word in keywords for word in ['night', 'evening', 'after dark']):
                    enhanced['usage_timing'] = 'night'
                elif any(word in keywords for word in ['day', 'morning', 'afternoon']):
                    enhanced['usage_timing'] = 'day'
                else:
                    enhanced['usage_timing'] = 'mixed'
            
            # Extract quantity from NLP
            if 'entities' in nlp_analysis:
                entities = nlp_analysis['entities']
                if 'numbers' in entities:
                    numbers = entities['numbers']
                    if numbers:
                        enhanced['quantity'] = numbers[0]
            
            enhanced_appliances.append(enhanced)
        
        return enhanced_appliances
 
    def _analyze_appliance_with_ai(self, appliance: Dict, user_query: str) -> ApplianceAnalysis:
        """Analyze appliance with AI intelligence"""
        try:
            appliance_name = appliance.get('name', '')
            usage_hours = appliance.get('usage_hours', 8)
            quantity = appliance.get('quantity', 1)
            
            # Fuzzy matching for appliance identification
            matched_appliance, confidence = self._fuzzy_match_appliance(appliance_name)
            
            if matched_appliance is None:
                # Fallback to generic appliance
                matched_appliance = self._create_generic_appliance(appliance_name)
                confidence = 0.5
            
            # Calculate energy consumption
            power_watts = matched_appliance.get('power_watts', 100)
            daily_energy_kwh = (power_watts * usage_hours * quantity) / 1000
            
            # Generate AI explanation
            ai_explanation = self._generate_ai_explanation(
                appliance_name, matched_appliance, usage_hours, quantity, user_query
            )
            
            # Identify usage pattern
            usage_pattern = self._identify_usage_pattern(usage_hours, appliance_name)
            
            # Generate energy tips
            energy_tips = self._generate_energy_tips(matched_appliance, usage_hours)
            
            # Suggest alternatives
            alternatives = self._suggest_alternatives(matched_appliance)
            
            return ApplianceAnalysis(
                appliance_name=appliance_name,
                matched_appliance=matched_appliance.get('name', appliance_name),
                power_watts=power_watts,
                usage_hours=usage_hours,
                quantity=quantity,
                daily_energy_kwh=daily_energy_kwh,
                confidence_score=confidence,
                ai_explanation=ai_explanation,
                usage_pattern=usage_pattern,
                energy_tips=energy_tips,
                alternatives=alternatives
            )
            
        except Exception as e:
            self.log_activity(f"Error analyzing appliance {appliance.get('name', '')}: {e}", 'warning')
            return self._create_fallback_analysis(appliance)
 
    def _fuzzy_match_appliance(self, appliance_name: str) -> Tuple[Optional[Dict], float]:
        """Fuzzy match appliance name to database"""
        try:
            if self.appliances_df is None:
                return None, 0.0
            
            # Try exact match first
            exact_match = self.appliances_df[
                self.appliances_df['Appliance'].str.contains(appliance_name, case=False, na=False)
            ]
            if not exact_match.empty:
                return exact_match.iloc[0].to_dict(), 1.0
            
            # Try fuzzy matching if available
            if FUZZYWUZZY_AVAILABLE:
                appliance_names = self.appliances_df['Appliance'].tolist()
                match_result = process.extractOne(appliance_name, appliance_names, scorer=fuzz.ratio)
                
                if match_result and match_result[1] >= 60:  # 60% similarity threshold
                    matched_name = match_result[0]
                    matched_appliance = self.appliances_df[
                        self.appliances_df['Appliance'] == matched_name
                    ].iloc[0].to_dict()
                    return matched_appliance, match_result[1] / 100.0
            else:
                # Basic string matching without fuzzywuzzy
                appliance_names = self.appliances_df['Appliance'].tolist()
                for name in appliance_names:
                    if appliance_name.lower() in name.lower() or name.lower() in appliance_name.lower():
                        matched_appliance = self.appliances_df[
                            self.appliances_df['Appliance'] == name
                        ].iloc[0].to_dict()
                        return matched_appliance, 0.7  # Lower confidence for basic matching
            
            # Try synonym matching
            for synonym, main_name in self.synonyms_db.items():
                if synonym.lower() in appliance_name.lower():
                    matched_appliance = self.appliances_df[
                        self.appliances_df['Appliance'] == main_name
                    ]
                    if not matched_appliance.empty:
                        return matched_appliance.iloc[0].to_dict(), 0.8
            
            return None, 0.0
            
        except Exception as e:
            self.log_activity(f"Error in fuzzy matching: {e}", 'warning')
            return None, 0.0
 
    def _generate_ai_explanation(self, appliance_name: str, matched_appliance: Dict, 
                                usage_hours: float, quantity: int, user_query: str) -> str:
        """Generate AI explanation for appliance analysis"""
        if not self.llm_manager:
            return f"Analyzed {appliance_name} with {usage_hours}h daily usage"
        
        llm = self.llm_manager.get_llm('generation')
        if not llm:
            return f"Analyzed {appliance_name} with {usage_hours}h daily usage"
        
        try:
            prompt = f"""
            Explain the energy analysis for this appliance:
            
            Appliance: {appliance_name}
            Matched: {matched_appliance.get('name', 'Unknown')}
            Power: {matched_appliance.get('power_watts', 100)}W
            Usage: {usage_hours} hours/day
            Quantity: {quantity}
            User Query: {user_query}
            
            Provide:
            1. Energy consumption explanation
            2. Usage pattern analysis
            3. Energy efficiency insights
            4. Recommendations for optimization
            """
            
            response = llm.invoke(prompt)
            return response
            
        except Exception as e:
            self.log_activity(f"Error generating AI explanation: {e}", 'warning')
            return f"Analyzed {appliance_name} with {usage_hours}h daily usage"
 
    def _identify_usage_pattern(self, usage_hours: float, appliance_name: str) -> str:
        """Identify usage pattern based on hours and appliance type"""
        if usage_hours >= 12:
            return "Heavy usage - continuous operation"
        elif usage_hours >= 8:
            return "Regular usage - daily operation"
        elif usage_hours >= 4:
            return "Moderate usage - intermittent operation"
        elif usage_hours >= 1:
            return "Light usage - occasional operation"
        else:
            return "Minimal usage - rare operation"
 
    def _generate_energy_tips(self, matched_appliance: Dict, usage_hours: float) -> List[str]:
        """Generate energy saving tips for appliance"""
        tips = []
        
        appliance_type = matched_appliance.get('Type', '').lower()
        power_watts = matched_appliance.get('power_watts', 100)
        
        # General tips
        if usage_hours > 8:
            tips.append("Consider reducing usage hours to save energy")
        
        if power_watts > 1000:
            tips.append("This is a high-power appliance - use during peak sun hours")
        
        # Appliance-specific tips
        if 'fan' in appliance_type:
            tips.append("Use ceiling fans instead of standing fans for better efficiency")
            tips.append("Clean fan blades regularly for optimal performance")
        elif 'fridge' in appliance_type:
            tips.append("Keep fridge temperature at 3-4°C for optimal efficiency")
            tips.append("Ensure proper ventilation around the fridge")
        elif 'tv' in appliance_type:
            tips.append("Use energy-saving mode when available")
            tips.append("Turn off completely when not in use")
        elif 'light' in appliance_type:
            tips.append("Consider LED alternatives for better efficiency")
            tips.append("Use natural light during the day")
        
        return tips
 
    def _suggest_alternatives(self, matched_appliance: Dict) -> List[str]:
        """Suggest energy-efficient alternatives"""
        alternatives = []
        
        appliance_type = matched_appliance.get('Type', '').lower()
        power_watts = matched_appliance.get('power_watts', 100)
        
        if 'fan' in appliance_type:
            alternatives.append("Ceiling fan (more efficient)")
            alternatives.append("DC ceiling fan (lowest power consumption)")
        elif 'fridge' in appliance_type:
            alternatives.append("Energy Star rated fridge")
            alternatives.append("Inverter fridge (variable speed)")
        elif 'light' in appliance_type:
            alternatives.append("LED bulbs (80% less power)")
            alternatives.append("Solar-powered lights")
        elif 'tv' in appliance_type:
            alternatives.append("LED TV (more efficient)")
            alternatives.append("Smaller screen size")
        
        return alternatives
 
    def _generate_system_insights(self, appliance_analyses: List[ApplianceAnalysis], 
                                 total_energy: float) -> Dict[str, Any]:
        """Generate system-level insights"""
        try:
            # Calculate insights
            high_consumption = [a for a in appliance_analyses if a.daily_energy_kwh > 2.0]
            low_consumption = [a for a in appliance_analyses if a.daily_energy_kwh < 0.5]
            
            # Peak usage analysis
            peak_appliances = [a for a in appliance_analyses if a.usage_hours >= 8]
            
            # Energy distribution
            energy_distribution = {}
            for analysis in appliance_analyses:
                appliance_type = analysis.matched_appliance
                if appliance_type not in energy_distribution:
                    energy_distribution[appliance_type] = 0
                energy_distribution[appliance_type] += analysis.daily_energy_kwh
            
            # Generate recommendations
            recommendations = []
            if total_energy > 10:
                recommendations.append("High energy consumption - consider system oversizing")
            if len(high_consumption) > 2:
                recommendations.append("Multiple high-consumption appliances - plan for peak loads")
            if len(peak_appliances) > 3:
                recommendations.append("Many appliances run continuously - ensure adequate battery capacity")
            
            return {
                'total_energy_kwh': total_energy,
                'appliance_count': len(appliance_analyses),
                'high_consumption_appliances': len(high_consumption),
                'low_consumption_appliances': len(low_consumption),
                'peak_usage_appliances': len(peak_appliances),
                'energy_distribution': energy_distribution,
                'recommendations': recommendations,
                'system_complexity': self._assess_system_complexity(total_energy, len(appliance_analyses))
            }
            
        except Exception as e:
            self.log_activity(f"Error generating system insights: {e}", 'warning')
            return {}
 
    def _assess_system_complexity(self, total_energy: float, appliance_count: int) -> str:
        """Assess system complexity based on energy and appliance count"""
        if total_energy > 15 and appliance_count > 8:
            return "High complexity - requires professional installation"
        elif total_energy > 8 and appliance_count > 5:
            return "Medium complexity - moderate installation requirements"
        else:
            return "Low complexity - suitable for DIY installation"
 
    def _create_generic_appliance(self, appliance_name: str) -> Dict[str, Any]:
        """Create generic appliance data when no match found"""
        return {
            'name': appliance_name,
            'power_watts': 100,  # Default power
            'Type': 'Generic',
            'Category': 'Unknown',
            'min_power_w': 50,
            'max_power_w': 150
        }
 
    def _create_fallback_analysis(self, appliance: Dict) -> ApplianceAnalysis:
        """Create fallback analysis when processing fails"""
        return ApplianceAnalysis(
            appliance_name=appliance.get('name', 'Unknown'),
            matched_appliance='Generic',
            power_watts=100,
            usage_hours=appliance.get('usage_hours', 8),
            quantity=appliance.get('quantity', 1),
            daily_energy_kwh=0.8,  # 100W * 8h / 1000
            confidence_score=0.5,
            ai_explanation="Generic analysis - specific appliance not found",
            usage_pattern="Unknown",
            energy_tips=["Consider energy-efficient alternatives"],
            alternatives=["Look for energy-efficient models"]
        )
 
    def _load_appliance_data(self):
        """Load appliance data from CSV"""
        try:
            if os.path.exists('data/interim/cleaned/appliances_cleaned.csv'):
                self.appliances_df = pd.read_csv('data/interim/cleaned/appliances_cleaned.csv')
                self.log_activity(f"Loaded {len(self.appliances_df)} appliances")
            else:
                self.log_activity("No appliance data file found", 'warning')
        except Exception as e:
            self.log_activity(f"Error loading appliance data: {e}", 'warning')
            self.appliances_df = None
 
    def _load_synonyms(self):
        """Load appliance synonyms database"""
        try:
            # Load synonyms from file if available
            if os.path.exists('backend/app/agents/synonyms_database.py'):
                # Import synonyms
                import sys
                sys.path.append('backend/app/agents')
                from synonyms_database import SMART_SYNONYMS
                self.synonyms_db = SMART_SYNONYMS
                self.log_activity(f"Loaded {len(self.synonyms_db)} synonyms")
            else:
                # Create basic synonyms
                self.synonyms_db = {
                    'fan': 'Standing Fan',
                    'fridge': 'Refrigerator',
                    'tv': 'Television',
                    'light': 'Light Bulb',
                    'bulb': 'Light Bulb'
                }
                self.log_activity("Using basic synonyms")
        except Exception as e:
            self.log_activity(f"Error loading synonyms: {e}", 'warning')
            self.synonyms_db = {}
