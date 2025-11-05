"""
Tool Manager for Solar Recommender System
Provides external API tools for agents
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Tool Manager for Solar Recommender System
    Provides access to external APIs and tools
    """
    
    def __init__(self):
        """Initialize the tool manager"""
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.geocoding_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
    def get_weather_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get weather data for a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dict containing weather data
        """
        try:
            if not self.weather_api_key:
                logger.warning("Weather API key not available")
                return self._get_default_weather_data()
                
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'cloudiness': data['clouds']['all'],
                'description': data['weather'][0]['description'],
                'location': data['name']
            }
            
            except Exception as e:
            logger.error(f"Weather API call failed: {e}")
            return self._get_default_weather_data()
            
    def _get_default_weather_data(self) -> Dict[str, Any]:
        """Get default weather data for Nigerian climate"""
        return {
            'success': True,
            'temperature': 28.0,  # Average Nigerian temperature
            'humidity': 70.0,
            'pressure': 1013.25,
            'wind_speed': 3.5,
            'cloudiness': 40.0,
            'description': 'partly cloudy',
            'location': 'Nigeria',
            'source': 'default'
        }
        
    def get_solar_irradiance(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get solar irradiance data for a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dict containing solar irradiance data
        """
        try:
            # For now, use estimated values based on Nigerian solar potential
            # In a real implementation, you would use a solar irradiance API
            
            # Nigerian solar potential varies by region
            if latitude > 10:  # Northern regions
                daily_irradiance = 5.5  # kWh/m²/day
                peak_sun_hours = 6.0
            elif latitude > 6:  # Central regions
                daily_irradiance = 5.0  # kWh/m²/day
                peak_sun_hours = 5.5
            else:  # Southern regions
                daily_irradiance = 4.5  # kWh/m²/day
                peak_sun_hours = 5.0
                
            return {
                'success': True,
                'daily_irradiance': daily_irradiance,
                'peak_sun_hours': peak_sun_hours,
                'latitude': latitude,
                'longitude': longitude,
                'source': 'estimated'
            }
            
        except Exception as e:
            logger.error(f"Solar irradiance calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'daily_irradiance': 5.0,
                'peak_sun_hours': 5.5,
                'source': 'fallback'
            }
            
    def get_location_data(self, city: str, state: str) -> Dict[str, Any]:
        """
        Get location data including coordinates
        
        Args:
            city: City name
            state: State name
            
        Returns:
            Dict containing location data
        """
        try:
            # Nigerian city coordinates (simplified)
            nigerian_cities = {
                'Lagos': {'lat': 6.5244, 'lng': 3.3792, 'region': 'South West'},
                'Abuja': {'lat': 9.0765, 'lng': 7.3986, 'region': 'North Central'},
                'Kano': {'lat': 12.0022, 'lng': 8.5920, 'region': 'North West'},
                'Ibadan': {'lat': 7.3776, 'lng': 3.9470, 'region': 'South West'},
                'Port Harcourt': {'lat': 4.8156, 'lng': 7.0498, 'region': 'South South'},
                'Kaduna': {'lat': 10.5264, 'lng': 7.4381, 'region': 'North West'},
                'Maiduguri': {'lat': 11.8333, 'lng': 13.1500, 'region': 'North East'},
                'Enugu': {'lat': 6.4413, 'lng': 7.4988, 'region': 'South East'},
                'Abeokuta': {'lat': 7.1557, 'lng': 3.3451, 'region': 'South West'},
                'Jos': {'lat': 9.9167, 'lng': 8.9000, 'region': 'North Central'}
            }
            
            # Try to find exact match first
            if city in nigerian_cities:
                return {
                    'success': True,
                    'city': city,
                    'state': state,
                    'latitude': nigerian_cities[city]['lat'],
                    'longitude': nigerian_cities[city]['lng'],
                    'region': nigerian_cities[city]['region']
                }
                
            # Fallback to state-based coordinates
            state_coordinates = {
                'Lagos': {'lat': 6.5244, 'lng': 3.3792, 'region': 'South West'},
                'Abuja': {'lat': 9.0765, 'lng': 7.3986, 'region': 'North Central'},
                'Kano': {'lat': 12.0022, 'lng': 8.5920, 'region': 'North West'},
                'Oyo': {'lat': 7.3776, 'lng': 3.9470, 'region': 'South West'},
                'Rivers': {'lat': 4.8156, 'lng': 7.0498, 'region': 'South South'},
                'Kaduna': {'lat': 10.5264, 'lng': 7.4381, 'region': 'North West'},
                'Borno': {'lat': 11.8333, 'lng': 13.1500, 'region': 'North East'},
                'Enugu': {'lat': 6.4413, 'lng': 7.4988, 'region': 'South East'},
                'Ogun': {'lat': 7.1557, 'lng': 3.3451, 'region': 'South West'},
                'Plateau': {'lat': 9.9167, 'lng': 8.9000, 'region': 'North Central'}
            }
            
            if state in state_coordinates:
                return {
                    'success': True,
                    'city': city,
                    'state': state,
                    'latitude': state_coordinates[state]['lat'],
                    'longitude': state_coordinates[state]['lng'],
                    'region': state_coordinates[state]['region']
                }
                
            # Ultimate fallback
            return {
                'success': True,
                'city': city,
                'state': state,
                'latitude': 9.0765,  # Abuja coordinates
                'longitude': 7.3986,
                'region': 'North Central',
                'source': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Location data retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'city': city,
                'state': state,
                'latitude': 9.0765,
                'longitude': 7.3986,
                'region': 'North Central',
                'source': 'error_fallback'
            }
            
    def calculate_solar_potential(self, latitude: float, longitude: float, 
                                 weather_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate solar potential for a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            weather_data: Optional weather data
            
        Returns:
            Dict containing solar potential calculations
        """
        try:
            # Get solar irradiance data
            irradiance_data = self.get_solar_irradiance(latitude, longitude)
            
            if not irradiance_data['success']:
                raise Exception("Failed to get solar irradiance data")
                
            daily_irradiance = irradiance_data['daily_irradiance']
            peak_sun_hours = irradiance_data['peak_sun_hours']
            
            # Adjust for weather conditions if available
            if weather_data and weather_data.get('success'):
                cloudiness = weather_data.get('cloudiness', 50)
                # Reduce irradiance based on cloudiness
                daily_irradiance *= (1 - cloudiness / 200)  # Simple adjustment
                peak_sun_hours *= (1 - cloudiness / 200)
                
            # Calculate monthly and yearly potential
            monthly_irradiance = daily_irradiance * 30
            yearly_irradiance = daily_irradiance * 365
            
        return {
                'success': True,
                'daily_irradiance': round(daily_irradiance, 2),
                'monthly_irradiance': round(monthly_irradiance, 2),
                'yearly_irradiance': round(yearly_irradiance, 2),
                'peak_sun_hours': round(peak_sun_hours, 2),
                'latitude': latitude,
                'longitude': longitude,
                'solar_potential': 'High' if daily_irradiance > 5.0 else 'Medium' if daily_irradiance > 4.0 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Solar potential calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'daily_irradiance': 5.0,
                'peak_sun_hours': 5.5,
                'solar_potential': 'Medium'
            }
            
    def get_tool_status(self) -> Dict[str, Any]:
        """Get the status of all tools"""
        return {
            'weather_api': bool(self.weather_api_key),
            'geocoding_api': bool(self.geocoding_api_key),
            'tools_available': ['weather', 'solar_irradiance', 'location', 'solar_potential']
        }
