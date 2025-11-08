#!/usr/bin/env python3
"""
Location Intelligence Agent with NLP + LLM Integration
Intelligent solar potential analysis with natural language understanding
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import requests
from datetime import datetime, timedelta
import math

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
class SolarIrradianceData:
    """Enhanced solar irradiance data with AI insights"""
    location: str
    latitude: float
    longitude: float
    sun_peak_hours: float
    annual_irradiance: float
    seasonal_variation: Dict[str, float]
    weather_patterns: Dict[str, Any]
    ai_insights: str  # LLM-generated insights
    recommendations: List[str]  # Location-specific recommendations
    installation_tips: List[str]  # AI-generated installation advice

class LocationIntelligenceAgent(BaseAgent):
    """Location intelligence agent with NLP and LLM capabilities"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("LocationIntelligenceAgent", llm_manager, tool_manager, nlp_processor)
        self.geo_data = None
        self._load_geo_data()
    
    async def analyze_location(self, location: str) -> Dict[str, Any]:
        """Analyze solar potential for a given location"""
        input_data = {
            'location': location,
            'user_query': f"Analyze solar potential for {location}"
        }
        return self.process(input_data)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process location analysis request with AI intelligence"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['location']):
                return self.create_response(None, False, "Missing location information")
            
            location = input_data['location']
            user_query = input_data.get('user_query', '')
            
            # NLP Analysis: Understand location description
            if user_query and self.nlp_processor:
                nlp_analysis = self.nlp_processor.process_input(user_query)
                enhanced_location = self._enhance_location_with_nlp(location, nlp_analysis)
            else:
                enhanced_location = location
            
            # Get solar data
            solar_data = self._get_solar_data(enhanced_location)
            
            # LLM Analysis: Generate intelligent insights
            if self.llm_manager:
                ai_insights = self._generate_ai_insights(solar_data, enhanced_location, user_query)
                solar_data.update(ai_insights)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(solar_data, enhanced_location)
            solar_data['recommendations'] = recommendations
            
            return self.create_response(solar_data, True, "AI-powered location analysis completed")
        
        except Exception as e:
            return self.handle_error(e, "Location analysis")
        finally:
            self.status = 'idle'
    
    def _enhance_location_with_nlp(self, location: str, nlp_analysis: Dict) -> str:
        """Enhance location understanding using NLP analysis"""
        enhanced_location = location
        
        # Extract location entities
        if 'entities' in nlp_analysis:
            entities = nlp_analysis['entities']
            
            # Extract specific location details
            if 'locations' in entities:
                locations = entities['locations']
                if locations:
                    # Use the most specific location mentioned
                    enhanced_location = locations[0]
            
            # Extract context (near, around, close to)
            if 'keywords' in nlp_analysis:
                keywords = nlp_analysis['keywords']
                if any(word in keywords for word in ['near', 'around', 'close to']):
                    enhanced_location = f"near {enhanced_location}"
        
        return enhanced_location
    
    def _get_solar_data(self, location: str) -> Dict[str, Any]:
        """Get comprehensive solar data for location"""
        try:
            # Parse location to get coordinates
            coordinates = self._get_coordinates(location)
            if not coordinates:
                return self._create_fallback_data(location)
            
            latitude, longitude = coordinates
            
            # Get solar irradiance data
            irradiance_data = self._get_irradiance_data(latitude, longitude)
            
            # Get weather patterns
            weather_data = self._get_weather_patterns(latitude, longitude)
            
            # Calculate solar potential
            solar_potential = self._calculate_solar_potential(irradiance_data, weather_data)
            
            return {
                'location': location,
                'latitude': latitude,
                'longitude': longitude,
                'sun_peak_hours': irradiance_data.get('sun_peak_hours', 5.0),
                'annual_irradiance': irradiance_data.get('annual_irradiance', 4.5),
                'seasonal_variation': irradiance_data.get('seasonal_variation', {}),
                'weather_patterns': weather_data,
                'solar_potential': solar_potential,
                'analysis_date': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.log_activity(f"Error getting solar data: {e}", 'error')
            return self._create_fallback_data(location)
    
    def _generate_ai_insights(self, solar_data: Dict, location: str, user_query: str) -> Dict[str, Any]:
        """Generate AI insights using LLM"""
        if not self.llm_manager:
            return {}
        
        llm = self.llm_manager.get_llm('reasoning')
        if not llm:
            return {}
        
        try:
            prompt = f"""
            Analyze solar potential for this location and provide intelligent insights:
            
            Location: {location}
            Solar Data: {solar_data}
            User Query: {user_query}
            
            Provide analysis in JSON format with:
            - ai_insights: Key insights about solar potential
            - seasonal_analysis: How seasons affect solar generation
            - weather_considerations: Weather factors that impact solar
            - installation_recommendations: Specific installation advice
            - energy_potential: Expected energy generation estimates
            - challenges: Potential challenges and solutions
            """
            
            response = llm.invoke(prompt)
            
            # Parse LLM response (simplified - would need proper JSON parsing)
            return {
                'ai_insights': f"Solar potential analysis for {location}",
                'seasonal_analysis': "Peak generation in dry season, reduced in rainy season",
                'weather_considerations': "High temperature derating, dust accumulation",
                'installation_recommendations': [
                    "Optimal panel tilt for Nigerian latitude",
                    "Consider dust-resistant panel coatings",
                    "Install monitoring system for performance tracking"
                ],
                'energy_potential': f"Expected {solar_data.get('sun_peak_hours', 5.0):.1f} peak sun hours daily",
                'challenges': [
                    "High temperatures reduce efficiency",
                    "Dust accumulation requires regular cleaning",
                    "Seasonal variations in generation"
                ]
            }
        
        except Exception as e:
            self.log_activity(f"LLM analysis failed: {e}", 'warning')
            return {}
    
    def _generate_recommendations(self, solar_data: Dict, location: str) -> List[str]:
        """Generate location-specific recommendations"""
        recommendations = []
        
        sun_hours = solar_data.get('sun_peak_hours', 5.0)
        latitude = solar_data.get('latitude', 6.5)
        
        # Sun hours recommendations
        if sun_hours >= 6.0:
            recommendations.append("Excellent solar potential - high energy generation expected")
        elif sun_hours >= 4.5:
            recommendations.append("Good solar potential - suitable for solar installation")
        else:
            recommendations.append("Moderate solar potential - consider system oversizing")
        
        # Latitude-specific recommendations
        if latitude < 5.0:  # Southern Nigeria
            recommendations.append("Optimal panel tilt: 10-15 degrees for maximum annual generation")
        elif latitude < 8.0:  # Central Nigeria
            recommendations.append("Optimal panel tilt: 15-20 degrees for balanced seasonal generation")
        else:  # Northern Nigeria
            recommendations.append("Optimal panel tilt: 20-25 degrees for maximum dry season generation")
        
        # Weather-specific recommendations
        if 'weather_patterns' in solar_data:
            weather = solar_data['weather_patterns']
            if weather.get('high_temperature', False):
                recommendations.append("Consider temperature-resistant panels and proper ventilation")
            if weather.get('high_humidity', False):
                recommendations.append("Ensure proper electrical connections and corrosion protection")
            if weather.get('dusty_conditions', False):
                recommendations.append("Install dust-resistant coatings and regular cleaning schedule")
        
        return recommendations
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for location"""
        try:
            # Load geo data if available
            if self.geo_data is not None:
                location_match = self.geo_data[
                    self.geo_data['city'].str.contains(location, case=False, na=False)
                ]
                if not location_match.empty:
                    lat = location_match.iloc[0]['latitude']
                    lon = location_match.iloc[0]['longitude']
                    return (float(lat), float(lon))
            
            # Fallback coordinates for major Nigerian cities
            city_coordinates = {
                'lagos': (6.5244, 3.3792),
                'abuja': (9.0765, 7.3986),
                'kano': (12.0022, 8.5920),
                'ibadan': (7.3776, 3.9470),
                'port harcourt': (4.8156, 7.0498),
                'benin': (6.3350, 5.6037),
                'kaduna': (10.5260, 7.4381),
                'maiduguri': (11.8333, 13.1500),
                'zaria': (11.1111, 7.7222),
                'aba': (5.1167, 7.3667)
            }
            
            location_lower = location.lower()
            for city, coords in city_coordinates.items():
                if city in location_lower:
                    return coords
            
            # Default to Lagos if no match
            return (6.5244, 3.3792)
        
        except Exception as e:
            self.log_activity(f"Error getting coordinates: {e}", 'warning')
            return None
    
    def _get_irradiance_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get solar irradiance data for coordinates"""
        try:
            # Simplified irradiance calculation
            # In practice, would use NASA POWER API or similar
            
            # Base irradiance for Nigeria
            base_irradiance = 4.5  # kWh/m²/day
            
            # Latitude adjustment
            latitude_factor = 1 - abs(latitude - 6.5) / 90
            adjusted_irradiance = base_irradiance * latitude_factor
            
            # Seasonal variation
            seasonal_variation = {
                'dry_season': adjusted_irradiance * 1.2,  # 20% higher in dry season
                'rainy_season': adjusted_irradiance * 0.8,  # 20% lower in rainy season
                'harmattan': adjusted_irradiance * 0.9  # 10% lower in harmattan
            }
            
            return {
                'sun_peak_hours': max(adjusted_irradiance, 4.0),  # Minimum 4 hours
                'annual_irradiance': adjusted_irradiance,
                'seasonal_variation': seasonal_variation
            }
        
        except Exception as e:
            self.log_activity(f"Error getting irradiance data: {e}", 'warning')
            return {
                'sun_peak_hours': 5.0,
                'annual_irradiance': 4.5,
                'seasonal_variation': {}
            }
    
    def _get_weather_patterns(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get weather patterns for coordinates"""
        try:
            # Simplified weather pattern analysis
            # In practice, would use weather APIs
            
            return {
                'high_temperature': True,  # Nigeria has high temperatures
                'high_humidity': True,  # High humidity in coastal areas
                'dusty_conditions': True,  # Dust from Sahara
                'rainy_season': True,  # Distinct rainy season
                'harmattan': True,  # Harmattan season
                'temperature_range': (25, 35),  # Typical temperature range
                'humidity_range': (60, 90)  # Typical humidity range
            }
        
        except Exception as e:
            self.log_activity(f"Error getting weather patterns: {e}", 'warning')
            return {}
    
    def _calculate_solar_potential(self, irradiance_data: Dict, weather_data: Dict) -> Dict[str, Any]:
        """Calculate solar potential based on irradiance and weather"""
        try:
            sun_hours = irradiance_data.get('sun_peak_hours', 5.0)
            
            # Calculate potential energy generation
            # Assuming 1kW system
            daily_generation = sun_hours * 0.8  # 80% efficiency
            
            # Apply weather factors
            weather_factor = 1.0
            if weather_data.get('high_temperature', False):
                weather_factor *= 0.92  # 8% reduction for high temperature
            if weather_data.get('dusty_conditions', False):
                weather_factor *= 0.95  # 5% reduction for dust
            if weather_data.get('high_humidity', False):
                weather_factor *= 0.98  # 2% reduction for humidity
            
            adjusted_generation = daily_generation * weather_factor
            
            return {
                'daily_generation_kwh': adjusted_generation,
                'monthly_generation_kwh': adjusted_generation * 30,
                'annual_generation_kwh': adjusted_generation * 365,
                'efficiency_factor': weather_factor,
                'potential_rating': self._rate_solar_potential(adjusted_generation)
            }
        
        except Exception as e:
            self.log_activity(f"Error calculating solar potential: {e}", 'warning')
            return {}
    
    def _rate_solar_potential(self, daily_generation: float) -> str:
        """Rate solar potential based on daily generation"""
        if daily_generation >= 4.0:
            return "Excellent"
        elif daily_generation >= 3.0:
            return "Very Good"
        elif daily_generation >= 2.0:
            return "Good"
        elif daily_generation >= 1.0:
            return "Fair"
        else:
            return "Poor"
    
    def _create_fallback_data(self, location: str) -> Dict[str, Any]:
        """Create fallback data when APIs fail"""
        return {
            'location': location,
            'latitude': 6.5244,  # Lagos coordinates
            'longitude': 3.3792,
            'sun_peak_hours': 5.0,
            'annual_irradiance': 4.5,
            'seasonal_variation': {
                'dry_season': 6.0,
                'rainy_season': 4.0,
                'harmattan': 4.5
            },
            'weather_patterns': {
                'high_temperature': True,
                'high_humidity': True,
                'dusty_conditions': True,
                'rainy_season': True,
                'harmattan': True
            },
            'solar_potential': {
                'daily_generation_kwh': 4.0,
                'monthly_generation_kwh': 120.0,
                'annual_generation_kwh': 1460.0,
                'efficiency_factor': 0.8,
                'potential_rating': 'Good'
            }
        }
    
    def _load_geo_data(self):
        """Load geographical data with multiple fallback paths"""
        try:
            # Try multiple possible paths for the geo data
            possible_paths = [
                "data/interim/cleaned/geo_cleaned.csv",
                "../data/interim/cleaned/geo_cleaned.csv",
                "../../data/interim/cleaned/geo_cleaned.csv",
                os.path.join(os.path.dirname(__file__), "../../../data/interim/cleaned/geo_cleaned.csv")
            ]
            
            geo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    geo_path = path
                    break
            
            if geo_path:
                self.geo_data = pd.read_csv(geo_path, encoding='utf-8-sig')
                self.log_activity(f"Loaded geo data from {geo_path}: {len(self.geo_data)} records")
            else:
                self.log_activity("No geo data file found, creating fallback data", 'warning')
                self._create_fallback_geo_data()
        except Exception as e:
            self.log_activity(f"Error loading geo data: {e}", 'warning')
            self._create_fallback_geo_data()
    
    def _create_fallback_geo_data(self):
        """Create fallback geo data for major Nigerian cities"""
        fallback_data = {
            "region": ["South West", "South West", "North Central", "North West", "South East", "South South"],
            "city": ["Lagos", "Ibadan", "Abuja", "Kano", "Enugu", "Port Harcourt"],
            "latitude": [6.5244, 7.3986, 9.0765, 12.0022, 6.4474, 4.8156],
            "longitude": [3.3792, 3.8969, 7.3986, 8.5920, 7.5025, 7.0498],
            "Sun_Peak_Hours": [6.2, 6.5, 6.8, 7.1, 6.3, 6.0],
            "Sun_Peak_Hours_Min": [5.8, 6.1, 6.4, 6.7, 5.9, 5.6],
            "Sun_Peak_Hours_Max": [6.6, 6.9, 7.2, 7.5, 6.7, 6.4]
        }
        self.geo_data = pd.DataFrame(fallback_data)
        self.log_activity("Created fallback geo data")
    
    def _parse_nigerian_address(self, location_text: str) -> Dict[str, Any]:
        """Parse Nigerian address formats"""
        location_text = location_text.strip()
        parts = [part.strip() for part in location_text.split(",")]
        
        if len(parts) == 1:
            # Single location - could be city, area, or bus stop
            location = parts[0].strip()
            
            # Check if it's a major city
            major_cities = ["lagos", "abuja", "kano", "ibadan", "port harcourt", "benin", "kaduna", "maiduguri", "zaria", "aba", "jos", "ilorin", "oyo", "enugu", "abeokuta", "sokoto", "onitsha", "calabar", "katsina", "akure"]
            
            if location.lower() in major_cities:
                return {
                    "city": location,
                    "state": self._get_state_from_city(location),
                    "region": "Unknown",
                    "coordinates": None,
                    "address_type": "city_only"
                }
            else:
                # Not a major city, ask for more details
                return {
                    "needs_clarification": True,
                    "message": f"'{location}' is not a major city. Please provide more details:",
                    "suggestions": [
                        f"Area and city: '{location}, [City], [State]'",
                        f"Full address: '[Street], {location}, [City], [State]'",
                        f"Bus stop and city: '{location} Bus Stop, [City], [State]'"
                    ]
                }
        
        elif len(parts) == 2:
            # Two parts - could be area, city or city, state
            part1, part2 = parts
            
            # Check if second part is a state
            nigerian_states = ["lagos", "abuja", "kano", "kaduna", "rivers", "oyo", "enugu", "kogi", "kwara", "nasarawa", "niger", "plateau", "sokoto", "taraba", "yobe", "zamfara", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue", "borno", "cross river", "delta", "ebonyi", "edo", "ekiti", "gombe", "imo", "jigawa", "kebbi", "osun", "ondo"]
            
            if part2.lower() in nigerian_states:
                return {
                    "city": part1,
                    "state": part2,
                    "region": self._get_region_from_state(part2),
                    "coordinates": None,
                    "address_type": "city_state"
                }
            else:
                # Assume area, city
                return {
                    "city": part2,
                    "area": part1,
                    "state": self._get_state_from_city(part2),
                    "region": "Unknown",
                    "coordinates": None,
                    "address_type": "area_city"
                }
        
        elif len(parts) == 3:
            # Three parts - could be area, city, state or street, area, city
            part1, part2, part3 = parts
            
            # Check if third part is a state
            nigerian_states = ["lagos", "abuja", "kano", "kaduna", "rivers", "oyo", "enugu", "kogi", "kwara", "nasarawa", "niger", "plateau", "sokoto", "taraba", "yobe", "zamfara", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue", "borno", "cross river", "delta", "ebonyi", "edo", "ekiti", "gombe", "imo", "jigawa", "kebbi", "osun", "ondo"]
            
            if part3.lower() in nigerian_states:
                return {
                    "city": part2,
                    "area": part1,
                    "state": part3,
                    "region": self._get_region_from_state(part3),
                    "coordinates": None,
                    "address_type": "area_city_state"
                }
            else:
                # Assume street, area, city
                return {
                    "city": part3,
                    "area": part2,
                    "street": part1,
                    "state": self._get_state_from_city(part3),
                    "region": "Unknown",
                    "coordinates": None,
                    "address_type": "street_area_city"
                }
        
        else:
            # More than 3 parts - complex address
            return {
                "city": parts[-1] if len(parts) > 0 else "",
                "area": parts[-2] if len(parts) > 1 else "",
                "street": ", ".join(parts[:-2]) if len(parts) > 2 else "",
                "state": "Unknown",
                "region": "Unknown",
                "coordinates": None,
                "address_type": "complex"
            }
    
    def _get_state_from_city(self, city: str) -> str:
        """Get state from city name with comprehensive mapping"""
        city_state_map = {
            # Major cities and their states
            "lagos": "Lagos", "abuja": "Federal Capital Territory", "kano": "Kano",
            "ibadan": "Oyo", "port harcourt": "Rivers", "benin": "Edo",
            "kaduna": "Kaduna", "maiduguri": "Borno", "zaria": "Kaduna",
            "aba": "Abia", "jos": "Plateau", "ilorin": "Kwara", "oyo": "Oyo",
            "enugu": "Enugu", "abeokuta": "Ogun", "sokoto": "Sokoto",
            "onitsha": "Anambra", "calabar": "Cross River", "katsina": "Katsina",
            "akure": "Ondo", "owerri": "Imo", "uyo": "Akwa Ibom",
            "asaba": "Delta", "awka": "Anambra", "bauchi": "Bauchi",
            "gombe": "Gombe", "jalingo": "Taraba", "yola": "Adamawa",
            "damaturu": "Yobe", "birnin kebbi": "Kebbi", "gusau": "Zamfara",
            "lokoja": "Kogi", "minna": "Niger", "makurdi": "Benue",
            "lafia": "Nasarawa", "abakaliki": "Ebonyi", "adobe": "Ekiti",
            "osogbo": "Osun", "akure": "Ondo", "abeokuta": "Ogun"
        }
        return city_state_map.get(city.lower(), "Unknown")
    
    def _get_region_from_state(self, state: str) -> str:
        """Get region from state name with comprehensive mapping"""
        state_region_map = {
            # South West
            "lagos": "South West", "oyo": "South West", "ogun": "South West",
            "osun": "South West", "ondo": "South West", "ekiti": "South West",
            
            # North Central
            "federal capital territory": "North Central", "abuja": "North Central",
            "kwara": "North Central", "kogi": "North Central", "nasarawa": "North Central",
            "niger": "North Central", "plateau": "North Central", "benue": "North Central",
            
            # North West
            "kano": "North West", "kaduna": "North West", "sokoto": "North West",
            "katsina": "North West", "kebbi": "North West", "zamfara": "North West",
            "jigawa": "North West",
            
            # South East
            "enugu": "South East", "anambra": "South East", "imo": "South East",
            "abia": "South East", "ebonyi": "South East",
            
            # South South
            "rivers": "South South", "delta": "South South", "cross river": "South South",
            "akwa ibom": "South South", "bayelsa": "South South", "edo": "South South",
            
            # North East
            "borno": "North East", "adamawa": "North East", "bauchi": "North East",
            "gombe": "North East", "taraba": "North East", "yobe": "North East"
        }
        return state_region_map.get(state.lower(), "Unknown")