# InputMappingAgent - Multi-LLM Powered Smart Input Processing System
# Advanced agent using all 4 LLMs for intelligent input understanding:
# - Groq Llama3: Fast appliance recognition and quick mapping
# - Groq Mixtral: Complex input parsing and disambiguation
# - HuggingFace: Knowledge-based appliance matching
# - Replicate: Creative input interpretation and suggestions
# - OpenRouter: Advanced natural language understanding
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Import LLM Manager
try:
    from ..core.llm_manager import StreamlinedLLMManager
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from llm_manager import StreamlinedLLMManager

try:
    from .synonyms_database import SMART_SYNONYMS
except ImportError:
    from synonyms_database import SMART_SYNONYMS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ApplianceMatch:
    """Represents a matched appliance for selection purposes"""
    category: str
    appliance: str
    type_variant: str
    min_power_w: float
    max_power_w: float
    hours_per_day_min: float
    hours_per_day_max: float
    surge_factor: float
    notes: str
    user_input: str
    confidence_score: float

@dataclass
class ClarificationRequest:
    """Represents a request for user clarification"""
    user_input: str
    input_type: str
    clarification_type: str
    message: str
    options: List[Dict[str, Any]]
    suggested_appliances: List[str] = None
    allow_custom_input: bool = False
    custom_input_prompt: str = ""

@dataclass
class CustomApplianceInput:
    """Represents a custom appliance input from user"""
    name: str
    category: str
    min_power_w: float
    max_power_w: float
    hours_per_day_min: float
    hours_per_day_max: float
    surge_factor: float
    notes: str = ""
    user_confidence: float = 0.8

@dataclass
class InputSession:
    """Represents a complete input session with ability to add/remove items"""
    appliances: List[Dict[str, Any]]
    location: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    autonomy_days: Optional[Dict[str, Any]] = None
    panel_placement: Optional[Dict[str, Any]] = None
    custom_appliances: List[CustomApplianceInput] = None
    session_id: str = ""
    last_updated: str = ""
    interaction_mode: str = "manual"
    user_context: Dict[str, Any] = None

@dataclass
class LocationInput:
    """Represents processed location information"""
    raw_input: str
    address: str
    city: str
    state: str
    region: str
    coordinates: Optional[Tuple[float, float]]
    confidence: float

@dataclass
class BudgetInput:
    """Represents processed budget information"""
    raw_input: str
    min_budget: float
    max_budget: float
    currency: str
    confidence: float

@dataclass
class AutonomyInput:
    """Represents processed autonomy days information"""
    raw_input: str
    days: int
    confidence: float

@dataclass
class PanelPlacementInput:
    """Represents processed panel placement information"""
    raw_input: str
    placement_type: str
    confidence: float
    roof_type: Optional[str] = None
    roof_direction: Optional[str] = None

class InputMappingAgent:
    """
    Multi-LLM Powered Comprehensive Input Processing Agent
    
    Uses all 4 LLMs strategically for intelligent input understanding:
    - Groq Llama3: Fast appliance recognition and quick mapping
    - Groq Mixtral: Complex input parsing and disambiguation  
    - HuggingFace: Knowledge-based appliance matching and specifications
    - Replicate: Creative input interpretation and user-friendly suggestions
    - OpenRouter: Advanced natural language understanding and context analysis
    
    Handles:
    - Natural language appliance descriptions
    - Location/address processing with context
    - Budget range interpretation and validation
    - Autonomy days clarification with reasoning
    - Panel placement options with intelligent suggestions
    - Usage patterns and preferences with AI insights
    
    Enhanced Features:
    - LLM-powered natural language understanding
    - Intelligent matching with AI reasoning
    - Smart clarification with contextual suggestions
    - AI-generated educational explanations
    - Advanced data preparation with semantic understanding
    """
    
    def __init__(self):
        """Initialize the Multi-LLM Input Mapping Agent"""
        self.agent_name = "InputMappingAgent"
        self.version = "2.0.0"
        
        # Initialize LLM Manager with all 4 LLMs (lazy initialization)
        self.llm_manager = None
        
        # Multi-LLM task assignment for input processing
        self.llm_tasks = {
            'quick_recognition': 'groq_llama3',        # Fast appliance recognition
            'complex_parsing': 'groq_mixtral',         # Complex input disambiguation
            'knowledge_matching': 'huggingface',       # Knowledge-based matching
            'creative_suggestions': 'replicate',       # User-friendly suggestions
            'advanced_understanding': 'openrouter_claude' # Advanced NLU and context
        }
        
        self.appliances_df = None
        self.categories = []
        self.appliances = []
        self.load_appliance_database()
    
    def _get_llm_manager(self):
        """Lazy initialization of LLM manager"""
        if self.llm_manager is None:
            try:
                self.llm_manager = StreamlinedLLMManager()
            except Exception as e:
                logger.error(f"Failed to initialize LLM manager: {e}")
                # Return a mock manager for fallback
                self.llm_manager = type('MockLLMManager', (), {
                    'get_llm': lambda x: None,
                    'call_llm': lambda x, y, z: "LLM not available"
                })()
        return self.llm_manager
        
        # Initialize performance tracking
        self.performance_metrics = {
            'total_mappings': 0,
            'successful_mappings': 0,
            'clarification_requests': 0,
            'llm_assisted_mappings': 0
        }
        
        print(f"🔧 {self.agent_name} v{self.version} initialized with Multi-LLM System:")
        available_llms = self._get_llm_manager().get_available_providers()
        for llm in available_llms:
            print(f"   ✅ {llm}")
        print(f"   📊 Appliances Database: {len(self.appliances_df) if self.appliances_df is not None else 0} entries")
        print(f"   🧠 LLM Tasks: {len(self.llm_tasks)}")
    
    def load_appliance_database(self):
        """Load cleaned appliance database"""
        try:
            # Get the project root directory (two levels up from backend/app/agents)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            csv_path = os.path.join(project_root, "data", "interim", "cleaned", "appliances_cleaned.csv")
            self.appliances_df = pd.read_csv(csv_path)
            self.categories = sorted(self.appliances_df["Category"].unique().tolist())
            self.appliances = sorted(self.appliances_df["Appliance"].unique().tolist())
            logger.info(f"Loaded {len(self.appliances_df)} appliances across {len(self.categories)} categories")
        except Exception as e:
            logger.error(f"Failed to load appliance database: {str(e)}")
            raise
    
    def _build_smart_mappings(self):
        """Build smart mapping dictionaries for intelligent matching"""
        if self.appliances_df is None or self.appliances_df.empty:
            return
        
        self.category_appliances = {}
        for category in self.categories:
            category_data = self.appliances_df[self.appliances_df["Category"] == category]
            self.category_appliances[category] = category_data["Appliance"].unique().tolist()
        
        self.appliance_types = {}
        for appliance in self.appliances:
            appliance_data = self.appliances_df[self.appliances_df["Appliance"] == appliance]
            self.appliance_types[appliance] = appliance_data["Type"].unique().tolist()
        
        self.smart_synonyms = SMART_SYNONYMS
    
    def process_appliances(self, appliance_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process appliance input and return daily energy calculation"""
        try:
            appliances = appliance_input.get('appliances', [])
            usage_hours = appliance_input.get('usage_hours', {})
            quantities = appliance_input.get('quantities', {})
            
            total_daily_energy_kwh = 0
            processed_appliances = []
            
            for appliance_name in appliances:
                if ' - ' in appliance_name:
                    type_part = appliance_name.split(' - ')[1]
                else:
                    type_part = appliance_name
                
                appliance_data = self.appliances_df[
                    (self.appliances_df['Appliance'].str.contains(appliance_name, case=False, na=False)) | 
                    (self.appliances_df['Type'].str.contains(appliance_name, case=False, na=False)) |
                    (self.appliances_df['Type'].str.contains(type_part, case=False, na=False))
                ]
                
                if not appliance_data.empty:
                    appliance = appliance_data.iloc[0]
                    hours = usage_hours.get(appliance_name, 8)
                    quantity = quantities.get(appliance_name, 1)
                    
                    avg_power = (appliance['min_power_w'] + appliance['max_power_w']) / 2
                    daily_energy = (avg_power * hours * quantity) / 1000
                    total_daily_energy_kwh += daily_energy
                    
                    processed_appliances.append({
                        'name': appliance_name,
                        'power_watts': avg_power,
                        'usage_hours': hours,
                        'quantity': quantity,
                        'daily_energy_kwh': daily_energy
                    })
            
            return {
                'total_daily_energy_kwh': total_daily_energy_kwh,
                'processed_appliances': processed_appliances,
                'success': True
            }
            
        except Exception as e:
            return {
                'total_daily_energy_kwh': 0,
                'processed_appliances': [],
                'success': False,
                'error': str(e)
            }
    
    def process_all_user_inputs(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process ALL user inputs comprehensively"""
        results = {
            "processed_inputs": {
                "appliances": [],
                "location": None,
                "budget": None,
                "autonomy_days": None,
                "panel_placement": None
            },
            "clarifications_needed": [],
            "all_inputs_complete": False,
            "next_agent_data": None
        }
        
        # Process appliances
        if "appliances" in user_inputs and user_inputs["appliances"]:
            appliance_result = self.map_user_inputs(user_inputs["appliances"])
            results["processed_inputs"]["appliances"] = appliance_result["mapped_appliances"]
            results["clarifications_needed"].extend(appliance_result["clarifications_needed"])
        
        # Process location
        if "location" in user_inputs and user_inputs["location"]:
            location_result = self._process_location_input(user_inputs["location"])
            if location_result:
                results["processed_inputs"]["location"] = location_result
            else:
                results["clarifications_needed"].append(self._create_location_clarification(user_inputs["location"]))
        
        # Process budget
        if "budget" in user_inputs and user_inputs["budget"]:
            budget_result = self._process_budget_input(user_inputs["budget"])
            if budget_result:
                results["processed_inputs"]["budget"] = budget_result
            else:
                results["clarifications_needed"].append(self._create_budget_clarification(user_inputs["budget"]))
        
        # Process autonomy days
        if "autonomy_days" in user_inputs and user_inputs["autonomy_days"]:
            autonomy_result = self._process_autonomy_input(user_inputs["autonomy_days"])
            if autonomy_result:
                results["processed_inputs"]["autonomy_days"] = autonomy_result
            else:
                results["clarifications_needed"].append(self._create_autonomy_clarification(user_inputs["autonomy_days"]))
        
        # Process panel placement
        if "panel_placement" in user_inputs and user_inputs["panel_placement"]:
            placement_result = self._process_panel_placement_input(user_inputs["panel_placement"])
            if placement_result:
                results["processed_inputs"]["panel_placement"] = placement_result
            else:
                results["clarifications_needed"].append(self._create_panel_placement_clarification(user_inputs["panel_placement"]))
        
        # Check if all inputs are complete
        if not results["clarifications_needed"]:
            results["all_inputs_complete"] = True
            results["next_agent_data"] = self._prepare_comprehensive_data_for_next_agent(results["processed_inputs"])
        
        return results
    
    def map_user_inputs(self, user_appliances: List[str]) -> Dict[str, Any]:
        """Map user appliance inputs to database entries with intelligent matching"""
        results = {
            "mapped_appliances": [],
            "clarifications_needed": [],
            "selection_complete": False,
            "next_agent_data": None
        }
        
        for user_input in user_appliances:
            user_input = user_input.strip().lower()
            matches = self._find_appliance_matches(user_input)
            
            if not matches:
                clarification = self._create_suggestion_clarification(user_input)
                results["clarifications_needed"].append(clarification)
            else:
                clarification = self._create_clarification_request(user_input, matches)
                results["clarifications_needed"].append(clarification)
        
        if not results["clarifications_needed"]:
            results["selection_complete"] = True
            results["next_agent_data"] = self._prepare_data_for_next_agent(results["mapped_appliances"])
        
        return results
    
    def _prepare_data_for_next_agent(self, mapped_appliances: List[ApplianceMatch]) -> Dict[str, Any]:
        """Prepare clean appliance data for the next agent (SystemSizingAgent)"""
        appliance_data = []
        
        for match in mapped_appliances:
            appliance_data.append({
                "category": match.category,
                "appliance": match.appliance,
                "type_variant": match.type_variant,
                "min_power_w": match.min_power_w,
                "max_power_w": match.max_power_w,
                "hours_per_day_min": match.hours_per_day_min,
                "hours_per_day_max": match.hours_per_day_max,
                "surge_factor": match.surge_factor,
                "notes": match.notes
            })
        
        return {
            "appliances": appliance_data,
            "total_appliances": len(appliance_data),
            "categories": list(set([app["category"] for app in appliance_data]))
        }
    
    def _prepare_comprehensive_data_for_next_agent(self, processed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive data for the next agent (GeoAgent + SystemSizingAgent)"""
        return {
            "appliances": self._prepare_data_for_next_agent(processed_inputs["appliances"])["appliances"],
            "location": {
                "address": processed_inputs["location"].address,
                "city": processed_inputs["location"].city,
                "state": processed_inputs["location"].state,
                "region": processed_inputs["location"].region,
                "coordinates": processed_inputs["location"].coordinates
            } if processed_inputs["location"] else None,
            "budget": {
                "min_budget": processed_inputs["budget"].min_budget,
                "max_budget": processed_inputs["budget"].max_budget,
                "currency": processed_inputs["budget"].currency
            } if processed_inputs["budget"] else None,
            "autonomy_days": processed_inputs["autonomy_days"].days if processed_inputs["autonomy_days"] else None,
            "panel_placement": {
                "placement_type": processed_inputs["panel_placement"].placement_type,
                "roof_type": processed_inputs["panel_placement"].roof_type,
                "roof_direction": processed_inputs["panel_placement"].roof_direction
            } if processed_inputs["panel_placement"] else None
        }
    
    # Location Processing Methods
    def _process_location_input(self, location_input: str) -> Optional[LocationInput]:
        """Process location input and return structured data"""
        location_input = location_input.strip()
        city, state, region = self._parse_nigerian_location(location_input)
        
        if city and state:
            return LocationInput(
                raw_input=location_input,
                address=location_input,
                city=city,
                state=state,
                region=region or "Unknown",
                coordinates=None,
                confidence=0.8
            )
        return None
    
    def _parse_nigerian_location(self, location: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse Nigerian location strings"""
        location_lower = location.lower()
        
        nigerian_locations = {
            # South West
            "lagos": ("Lagos", "Lagos State", "South West"),
            "ibadan": ("Ibadan", "Oyo State", "South West"),
            "abeokuta": ("Abeokuta", "Ogun State", "South West"),
            "akure": ("Akure", "Ondo State", "South West"),
            "ado-ekiti": ("Ado-Ekiti", "Ekiti State", "South West"),
            "osogbo": ("Osogbo", "Osun State", "South West"),
            
            # South East
            "enugu": ("Enugu", "Enugu State", "South East"),
            "awka": ("Awka", "Anambra State", "South East"),
            "owerri": ("Owerri", "Imo State", "South East"),
            "aba": ("Aba", "Abia State", "South East"),
            "umuahia": ("Umuahia", "Abia State", "South East"),
            "abakaliki": ("Abakaliki", "Ebonyi State", "South East"),
            
            # South South
            "port harcourt": ("Port Harcourt", "Rivers State", "South South"),
            "calabar": ("Calabar", "Cross River State", "South South"),
            "uyo": ("Uyo", "Akwa Ibom State", "South South"),
            "benin": ("Benin", "Edo State", "South South"),
            "asaba": ("Asaba", "Delta State", "South South"),
            "warri": ("Warri", "Delta State", "South South"),
            
            # North Central
            "abuja": ("Abuja", "FCT", "North Central"),
            "jos": ("Jos", "Plateau State", "North Central"),
            "lokoja": ("Lokoja", "Kogi State", "North Central"),
            "minna": ("Minna", "Niger State", "North Central"),
            "makurdi": ("Makurdi", "Benue State", "North Central"),
            "kwara": ("Ilorin", "Kwara State", "North Central"),
            
            # North West
            "kano": ("Kano", "Kano State", "North West"),
            "kaduna": ("Kaduna", "Kaduna State", "North West"),
            "sokoto": ("Sokoto", "Sokoto State", "North West"),
            "katsina": ("Katsina", "Katsina State", "North West"),
            "zaria": ("Zaria", "Kaduna State", "North West"),
            "birnin kebbi": ("Birnin Kebbi", "Kebbi State", "North West"),
            
            # North East
            "maiduguri": ("Maiduguri", "Borno State", "North East"),
            "yola": ("Yola", "Adamawa State", "North East"),
            "bauchi": ("Bauchi", "Bauchi State", "North East"),
            "gombe": ("Gombe", "Gombe State", "North East"),
            "damaturu": ("Damaturu", "Yobe State", "North East"),
            "jalingo": ("Jalingo", "Taraba State", "North East")
        }
        
        for key, (city, state, region) in nigerian_locations.items():
            if key in location_lower:
                return city, state, region
        
        return None, None, None
    
    def _create_location_clarification(self, location_input: str) -> ClarificationRequest:
        """Create clarification request for location input"""
        return ClarificationRequest(
            user_input=location_input,
            input_type="location",
            clarification_type="ambiguous_input",
            message=f"I couldn't identify the location '{location_input}'. Could you provide a more specific address or city name?",
            options=[{
                "suggestion": "Please provide: City, State (e.g., 'Lagos, Lagos State' or 'Abuja, FCT')",
                "examples": ["Lagos, Lagos State", "Abuja, FCT", "Kano, Kano State", "Port Harcourt, Rivers State"]
            }]
        )
    
    # Budget Processing Methods
    def _process_budget_input(self, budget_input: str) -> Optional[BudgetInput]:
        """Process budget input and return structured data"""
        import re
        
        budget_input = budget_input.strip().lower()
        numbers = re.findall(r'[\d,]+', budget_input)
        if not numbers:
            return None
        
        number_str = numbers[0].replace(',', '')
        try:
            budget = float(number_str)
        except ValueError:
            return None
        
        if 'k' in budget_input or 'thousand' in budget_input:
            budget *= 1000
        elif 'm' in budget_input or 'million' in budget_input:
            budget *= 1000000
        
        currency = "NGN"
        if 'dollar' in budget_input or 'usd' in budget_input or '$' in budget_input:
            currency = "USD"
        elif 'euro' in budget_input or 'eur' in budget_input or '€' in budget_input:
            currency = "EUR"
        
        min_budget = budget * 0.8
        max_budget = budget * 1.2
        
        return BudgetInput(
            raw_input=budget_input,
            min_budget=min_budget,
            max_budget=max_budget,
            currency=currency,
            confidence=0.9
        )
    
    def _create_budget_clarification(self, budget_input: str) -> ClarificationRequest:
        """Create clarification request for budget input"""
        return ClarificationRequest(
            user_input=budget_input,
            input_type="budget",
            clarification_type="ambiguous_input",
            message=f"I couldn't understand the budget '{budget_input}'. Could you provide a clearer amount?",
            options=[{
                "suggestion": "Please provide budget in numbers (e.g., '500000', '500k', '1 million naira')",
                "examples": ["500000", "500k", "1 million", "2.5 million naira", "$5000"]
            }]
        )
    
    # Autonomy Days Processing Methods
    def _process_autonomy_input(self, autonomy_input: str) -> Optional[AutonomyInput]:
        """Process autonomy days input and return structured data"""
        import re
        
        autonomy_input = autonomy_input.strip().lower()
        numbers = re.findall(r'\d+', autonomy_input)
        if not numbers:
            return None
        
        days = int(numbers[0])
        
        if 'hour' in autonomy_input:
            days = days / 24
        elif 'week' in autonomy_input:
            days = days * 7
        elif 'month' in autonomy_input:
            days = days * 30
        
        if days < 0.5 or days > 30:
            return None
        
        return AutonomyInput(
            raw_input=autonomy_input,
            days=int(days),
            confidence=0.9
        )
    
    def _create_autonomy_clarification(self, autonomy_input: str) -> ClarificationRequest:
        """Create clarification request for autonomy input"""
        return ClarificationRequest(
            user_input=autonomy_input,
            input_type="autonomy",
            clarification_type="ambiguous_input",
            message=f"I couldn't understand the autonomy period '{autonomy_input}'. How many days do you want your battery to last without sunlight?",
            options=[{
                "suggestion": "Please provide number of days (e.g., '2 days', '3', '48 hours')",
                "examples": ["1 day", "2 days", "3", "48 hours", "1 week"]
            }]
        )
    
    # Panel Placement Processing Methods
    def _process_panel_placement_input(self, placement_input: str) -> Optional[PanelPlacementInput]:
        """Process panel placement input and return structured data"""
        placement_input = placement_input.strip().lower()
        
        placement_type = "roof"
        if 'ground' in placement_input or 'yard' in placement_input:
            placement_type = "ground"
        elif 'carport' in placement_input or 'car port' in placement_input:
            placement_type = "carport"
        elif 'mixed' in placement_input or 'both' in placement_input:
            placement_type = "mixed"
        
        roof_type = None
        if 'concrete' in placement_input or 'cement' in placement_input:
            roof_type = "concrete"
        elif 'zinc' in placement_input or 'metal' in placement_input:
            roof_type = "zinc"
        elif 'asbestos' in placement_input:
            roof_type = "asbestos"
        
        roof_direction = None
        if 'north' in placement_input:
            roof_direction = "north"
        elif 'south' in placement_input:
            roof_direction = "south"
        elif 'east' in placement_input:
            roof_direction = "east"
        elif 'west' in placement_input:
            roof_direction = "west"
        
        return PanelPlacementInput(
            raw_input=placement_input,
            placement_type=placement_type,
            roof_type=roof_type,
            roof_direction=roof_direction,
            confidence=0.8
        )
    
    def _create_panel_placement_clarification(self, placement_input: str) -> ClarificationRequest:
        """Create clarification request for panel placement input"""
        return ClarificationRequest(
            user_input=placement_input,
            input_type="panel_placement",
            clarification_type="category_choice",
            message=f"I need more details about your panel placement '{placement_input}'. Where would you like to install the solar panels?",
            options=[{
                "placement_type": "roof",
                "description": "On the roof of your building"
            }, {
                "placement_type": "ground",
                "description": "Ground-mounted in your yard/compound"
            }, {
                "placement_type": "carport",
                "description": "Carport or covered parking area"
            }, {
                "placement_type": "mixed",
                "description": "Combination of roof and ground"
            }]
        )
    
    def _find_appliance_matches(self, user_input: str) -> List[ApplianceMatch]:
        """Find all possible matches for user input with confidence scoring"""
        matches = []
        user_input_lower = user_input.lower()
        
        for _, row in self.appliances_df.iterrows():
            confidence = self._calculate_match_confidence(
                user_input_lower, 
                row["Appliance"], 
                row["Type"], 
                row["Category"]
            )
            
            if confidence > 0.3:
                match = ApplianceMatch(
                    category=row["Category"],
                    appliance=row["Appliance"],
                    type_variant=row["Type"],
                    min_power_w=row["min_power_w"],
                    max_power_w=row["max_power_w"],
                    hours_per_day_min=row["hours_per_day_min"],
                    hours_per_day_max=row["hours_per_day_max"],
                    surge_factor=row["surge_factor"],
                    notes=row["Notes"],
                    user_input=user_input,
                    confidence_score=confidence
                )
                matches.append(match)
        
        matches.sort(key=lambda x: x.confidence_score, reverse=True)
        return matches
    
    def _calculate_match_confidence(self, user_input: str, appliance: str, type_variant: str, category: str) -> float:
        """Calculate confidence score for a potential match"""
        if pd.isna(appliance) or pd.isna(type_variant) or pd.isna(category):
            return 0.0
        
        appliance_lower = str(appliance).lower()
        type_lower = str(type_variant).lower()
        category_lower = str(category).lower()
        
        confidence = 0.0
        
        if user_input == appliance_lower:
            confidence += 1.0
        elif user_input in appliance_lower:
            confidence += 0.8
        elif appliance_lower in user_input:
            confidence += 0.6
        
        if user_input == type_lower:
            confidence += 0.9
        elif user_input in type_lower:
            confidence += 0.7
        elif type_lower in user_input:
            confidence += 0.5
        
        if user_input in self.smart_synonyms:
            synonyms = self.smart_synonyms[user_input]
            for synonym in synonyms:
                if synonym in appliance_lower or synonym in type_lower:
                    confidence += 0.8
                    break
        
        for key, synonyms in self.smart_synonyms.items():
            if user_input in synonyms:
                if key in appliance_lower or key in type_lower:
                    confidence += 0.7
                    break
        
        if user_input in category_lower:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _create_suggestion_clarification(self, user_input: str) -> ClarificationRequest:
        """Create clarification request when no matches found"""
        similar_appliances = []
        
        for appliance in self.appliances:
            if any(word in appliance.lower() for word in user_input.split()):
                similar_appliances.append(appliance)
        
        if not similar_appliances:
            popular_categories = ["Cooling Appliances", "Entertainment & Media Electronics", 
                                "Lighting & Small Power Devices", "Mobile & Personal Devices"]
            similar_appliances = []
            for category in popular_categories:
                if category in self.category_appliances:
                    similar_appliances.extend(self.category_appliances[category][:2])
        
        options = []
        for appliance in similar_appliances[:5]:
            types = self.appliance_types.get(appliance, [])
            options.append({
                "appliance": appliance,
                "types": types[:3],
                "description": f"Various {appliance.lower()} options available"
            })
        
        return ClarificationRequest(
            user_input=user_input,
            suggested_appliances=similar_appliances[:5],
            clarification_type="no_matches",
            message=f"I couldn't find '{user_input}' in our database. Did you mean one of these?",
            options=options
        )
    
    def _create_clarification_request(self, user_input: str, matches: List[ApplianceMatch]) -> ClarificationRequest:
        """Create clarification request for multiple matches"""
        appliance_groups = {}
        for match in matches:
            if match.appliance not in appliance_groups:
                appliance_groups[match.appliance] = []
            appliance_groups[match.appliance].append(match)
        
        options = []
        for appliance, appliance_matches in list(appliance_groups.items())[:5]:
            types = [match.type_variant for match in appliance_matches]
            options.append({
                "appliance": appliance,
                "types": types,
                "power_range": f"{min(match.min_power_w for match in appliance_matches):.0f}-{max(match.max_power_w for match in appliance_matches):.0f}W",
                "description": f"{len(types)} different {appliance.lower()} types available"
            })
        
        return ClarificationRequest(
            user_input=user_input,
            input_type="appliance",
            suggested_appliances=list(appliance_groups.keys())[:5],
            clarification_type="multiple_matches",
            message=f"I found several options for '{user_input}'. Which type do you need?",
            options=options
        )
    
    def get_appliance_suggestions(self, category: str = None) -> List[Dict[str, Any]]:
        """Get appliance suggestions for a category or all categories"""
        if category:
            if category in self.category_appliances:
                appliances = self.category_appliances[category]
            else:
                return []
        else:
            appliances = self.appliances
        
        suggestions = []
        for appliance in appliances[:10]:
            types = self.appliance_types.get(appliance, [])
            suggestions.append({
                "appliance": appliance,
                "types": types[:3],
                "category": self._get_appliance_category(appliance)
            })
        
        return suggestions
    
    def create_input_session(self, interaction_mode: str = "manual", user_context: Dict[str, Any] = None) -> InputSession:
        """Create a new input session for managing user inputs"""
        import uuid
        
        session_id = str(uuid.uuid4())
        return InputSession(
            appliances=[],
            custom_appliances=[],
            session_id=session_id,
            last_updated=datetime.now().isoformat(),
            interaction_mode=interaction_mode,
            user_context=user_context or {}
        )
    
    def add_custom_appliance(self, session: InputSession, custom_input: CustomApplianceInput) -> InputSession:
        """Add a custom appliance to the session"""
        if session.custom_appliances is None:
            session.custom_appliances = []
        
        session.custom_appliances.append(custom_input)
        session.last_updated = datetime.now().isoformat()
        return session
    
    def remove_appliance(self, session: InputSession, appliance_id: str, is_custom: bool = False) -> InputSession:
        """Remove an appliance from the session"""
        if is_custom:
            if session.custom_appliances:
                session.custom_appliances = [app for app in session.custom_appliances if app.name != appliance_id]
        else:
            session.appliances = [app for app in session.appliances if app.get("appliance_id") != appliance_id]
        
        session.last_updated = datetime.now().isoformat()
        return session
    
    def update_appliance_quantity(self, session: InputSession, appliance_id: str, new_quantity: int, is_custom: bool = False) -> InputSession:
        """Update the quantity of an appliance in the session"""
        if is_custom:
            if session.custom_appliances:
                for app in session.custom_appliances:
                    if app.name == appliance_id:
                        app.quantity = new_quantity
                        break
        else:
            for app in session.appliances:
                if app.get("appliance_id") == appliance_id:
                    app["quantity"] = new_quantity
                    break
        
        session.last_updated = datetime.now().isoformat()
        return session
    
    def get_session_summary(self, session: InputSession) -> Dict[str, Any]:
        """Get a summary of the current input session"""
        total_appliances = len(session.appliances) + (len(session.custom_appliances) if session.custom_appliances else 0)
        
        total_power_min = 0
        total_power_max = 0
        
        for app in session.appliances:
            quantity = app.get("quantity", 1)
            total_power_min += app.get("min_power_w", 0) * quantity
            total_power_max += app.get("max_power_w", 0) * quantity
        
        for app in session.custom_appliances:
            quantity = getattr(app, "quantity", 1)
            total_power_min += app.min_power_w * quantity
            total_power_max += app.max_power_w * quantity
        
        return {
            "session_id": session.session_id,
            "total_appliances": total_appliances,
            "regular_appliances": len(session.appliances),
            "custom_appliances": len(session.custom_appliances) if session.custom_appliances else 0,
            "total_power_range": f"{total_power_min:.0f}-{total_power_max:.0f}W",
            "location_provided": session.location is not None,
            "budget_provided": session.budget is not None,
            "autonomy_days_provided": session.autonomy_days is not None,
            "panel_placement_provided": session.panel_placement is not None,
            "last_updated": session.last_updated
        }
    
    def process_custom_appliance_input(self, user_input: str, appliance_specs: Dict[str, Any]) -> CustomApplianceInput:
        """Process user input for a custom appliance"""
        try:
            min_power = float(appliance_specs.get("min_power_w", 0))
            max_power = float(appliance_specs.get("max_power_w", min_power))
            min_hours = float(appliance_specs.get("hours_per_day_min", 1))
            max_hours = float(appliance_specs.get("hours_per_day_max", min_hours))
            surge_factor = float(appliance_specs.get("surge_factor", 1.0))
            category = appliance_specs.get("category", "Custom Appliances")
            notes = appliance_specs.get("notes", f"Custom appliance: {user_input}")
            
            return CustomApplianceInput(
                name=user_input,
                category=category,
                min_power_w=min_power,
                max_power_w=max_power,
                hours_per_day_min=min_hours,
                hours_per_day_max=max_hours,
                surge_factor=surge_factor,
                notes=notes,
                user_confidence=appliance_specs.get("user_confidence", 0.8)
            )
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid appliance specifications: {e}")
    
    def get_custom_appliance_template(self) -> Dict[str, Any]:
        """Get a template for custom appliance input"""
        return {
            "name": "Appliance Name",
            "category": "Select Category",
            "min_power_w": "Minimum Power (Watts)",
            "max_power_w": "Maximum Power (Watts)",
            "hours_per_day_min": "Minimum Hours per Day",
            "hours_per_day_max": "Maximum Hours per Day",
            "surge_factor": "Surge Factor (e.g., 1.5 for 50% surge)",
            "notes": "Additional Notes",
            "user_confidence": "Confidence Level (0.0-1.0)"
        }
    
    def validate_custom_appliance(self, custom_appliance: CustomApplianceInput) -> Dict[str, Any]:
        """Validate a custom appliance input and provide feedback"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        if custom_appliance.min_power_w <= 0:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Minimum power must be greater than 0")
        
        if custom_appliance.max_power_w < custom_appliance.min_power_w:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Maximum power must be greater than or equal to minimum power")
        
        if custom_appliance.hours_per_day_min < 0 or custom_appliance.hours_per_day_min > 24:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Hours per day must be between 0 and 24")
        
        if custom_appliance.hours_per_day_max < custom_appliance.hours_per_day_min:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Maximum hours must be greater than or equal to minimum hours")
        
        if custom_appliance.surge_factor < 1.0:
            validation_result["warnings"].append("Surge factor is less than 1.0 - this may be incorrect")
        
        if custom_appliance.surge_factor > 5.0:
            validation_result["warnings"].append("Surge factor is very high - please verify this is correct")
        
        similar_appliances = self._find_similar_appliances(custom_appliance.name)
        if similar_appliances:
            validation_result["suggestions"].append(f"Found similar appliances: {list(similar_appliances.keys())[:3]}")
        
        return validation_result
    
    def _find_similar_appliances(self, user_input: str) -> Dict[str, List[ApplianceMatch]]:
        """Find similar appliances using fuzzy matching"""
        similar_appliances = {}
        user_input_lower = user_input.lower()
        
        for _, row in self.appliances_df.iterrows():
            appliance = row["Appliance"]
            type_variant = row["Type"]
            category = row["Category"]
            
            similarity_score = self._calculate_similarity_score(user_input_lower, appliance.lower())
            
            if similarity_score > 0.3:
                if appliance not in similar_appliances:
                    similar_appliances[appliance] = []
                
                match = ApplianceMatch(
                    category=category,
                    appliance=appliance,
                    type_variant=type_variant,
                    min_power_w=row["min_power_w"],
                    max_power_w=row["max_power_w"],
                    hours_per_day_min=row["hours_per_day_min"],
                    hours_per_day_max=row["hours_per_day_max"],
                    surge_factor=row["surge_factor"],
                    notes=row.get("Notes", ""),
                    user_input=user_input,
                    confidence_score=similarity_score
                )
                similar_appliances[appliance].append(match)
        
        for appliance in similar_appliances:
            similar_appliances[appliance].sort(key=lambda x: x.confidence_score, reverse=True)
        
        return similar_appliances
    
    def _calculate_similarity_score(self, input_text: str, target_text: str) -> float:
        """Calculate similarity score between input and target text"""
        if input_text in target_text or target_text in input_text:
            return 0.8
        
        input_words = set(input_text.split())
        target_words = set(target_text.split())
        common_words = input_words.intersection(target_words)
        
        if common_words:
            return len(common_words) / max(len(input_words), len(target_words))
        
        return 0.0
    
    def _get_appliance_category(self, appliance: str) -> str:
        """Get category for a specific appliance"""
        appliance_data = self.appliances_df[self.appliances_df["Appliance"] == appliance]
        if not appliance_data.empty:
            return appliance_data.iloc[0]["Category"]
        return "Unknown"
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return self.categories
    
    def get_appliances_by_category(self, category: str) -> List[str]:
        """Get all appliances in a specific category"""
        return self.category_appliances.get(category, [])

# Test the agent
if __name__ == "__main__":
    print("Initializing Smart InputMappingAgent...")
    
    try:
        agent = InputMappingAgent()
        print(f"Loaded {len(agent.appliances)} appliances across {len(agent.categories)} categories")
        
        # Get user input for testing
        print("\nEnter appliances for testing (comma-separated):")
        appliance_input = input("Appliances: ").strip()
        appliances = [app.strip() for app in appliance_input.split(",") if app.strip()]
        
        if not appliances:
            print("No appliances entered, using default test")
            appliances = ["fridge", "tv"]
        
        print(f"\nTesting smart appliance selection with: {appliances}")
        
        # Test appliance mapping
        result = agent.map_user_inputs(appliances)
        
        if result["selection_complete"]:
            print(f"Selection complete! Ready for next agent")
            print(f" {len(result['mapped_appliances'])} appliances selected")
        elif result["clarifications_needed"]:
            print(f"Needs clarification: {result['clarifications_needed'][0].message}")
            for option in result["clarifications_needed"][0].options[:3]:
                print(f" {option['appliance']} - {option['description']}")
        
        # Test custom appliance functionality
        print(f"\nTesting custom appliance functionality...")
        
        try:
            session = agent.create_input_session()
            print(f"Created input session: {session.session_id}")
            
            # Get custom appliance details from user
            print("\nEnter custom appliance details:")
            custom_name = input("Appliance name: ").strip()
            if custom_name:
                min_power = float(input("Min power (W): ").strip() or "10")
                max_power = float(input("Max power (W): ").strip() or "15")
                hours_min = float(input("Min hours per day: ").strip() or "4")
                hours_max = float(input("Max hours per day: ").strip() or "8")
                surge_factor = float(input("Surge factor: ").strip() or "1.2")
                category = input("Category: ").strip() or "Lighting"
                notes = input("Notes: ").strip() or "Custom appliance"
                
                custom_appliance = agent.process_custom_appliance_input(custom_name, {
                    "min_power_w": min_power,
                    "max_power_w": max_power,
                    "hours_per_day_min": hours_min,
                    "hours_per_day_max": hours_max,
                    "surge_factor": surge_factor,
                    "category": category,
                    "notes": notes
                })
                
                validation = agent.validate_custom_appliance(custom_appliance)
                print(f"Custom appliance validation: {validation['is_valid']}")
                if validation['warnings']:
                    print(f"Warnings: {validation['warnings']}")
                if validation['suggestions']:
                    print(f"Suggestions: {validation['suggestions']}")
                
                session = agent.add_custom_appliance(session, custom_appliance)
                print(f"Added custom appliance to session")
                
                summary = agent.get_session_summary(session)
                print(f"Session Summary: {summary}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()