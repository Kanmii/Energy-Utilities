"""Mock agents for testing purposes"""
from typing import Dict, Any, List, Optional
from datetime import datetime

class MockInputMappingAgent:
    """Mock InputMappingAgent for testing"""
    
    def __init__(self):
        self.initialized = True
        self.appliance_data = {
            'LED TV - 43 inch': {
                'category': 'Electronics',
                'type': '43 inch LED TV',
                'min_power_w': 80,
                'max_power_w': 120,
                'hours_per_day_min': 4,
                'hours_per_day_max': 8,
                'surge_factor': 1.2,
                'notes': 'Smart TV with LED display'
            },
            'Refrigerator - Medium Size': {
                'category': 'Appliances',
                'type': 'Double Door',
                'min_power_w': 120,
                'max_power_w': 180,
                'hours_per_day_min': 24,
                'hours_per_day_max': 24,
                'surge_factor': 3.0,
                'notes': 'Energy efficient double door refrigerator'
            }
        }
    
    def process_appliance_data(self, appliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process appliance input data"""
        appliances = appliance_data.get('appliances', [])
        processed_appliances = []
        
        for appliance in appliances:
            if appliance['name'] in self.appliance_data:
                data = self.appliance_data[appliance['name']]
                processed_appliances.append({
                    'name': appliance['name'],
                    'category': data['category'],
                    'type': data['type'],
                    'power_watts': (data['min_power_w'] + data['max_power_w']) / 2,
                    'usage_hours': appliance['usage_hours'],
                    'quantity': appliance['quantity'],
                    'surge_factor': data['surge_factor']
                })
        
        return {
            'appliances': processed_appliances,
            'success': True
        }
    
    def calculate_total_power(self, appliances: List[Dict[str, Any]]) -> float:
        """Calculate total daily power consumption in watt-hours"""
        total_power = 0
        for appliance in appliances:
            total_power += appliance['power_watts'] * appliance['usage_hours']
        return total_power
    
    def calculate_surge_power(self, appliances: List[Dict[str, Any]]) -> float:
        """Calculate total surge power requirement in watts"""
        total_surge = 0
        for appliance in appliances:
            total_surge += appliance['power_watts'] * appliance['surge_factor']
        return total_surge
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure"""
        if not isinstance(input_data, dict) or 'appliances' not in input_data:
            raise ValueError("Input data must be a dictionary with 'appliances' key")
        
        for appliance in input_data['appliances']:
            required_fields = ['name', 'category', 'type', 'usage_hours', 'quantity', 'power_watts']
            if not all(field in appliance for field in required_fields):
                raise ValueError(f"Missing required fields in appliance data: {required_fields}")
        
        return True
    
    def generate_appliance_summary(self, appliances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of processed appliances"""
        total_daily = self.calculate_total_power(appliances)
        peak_surge = self.calculate_surge_power(appliances)
        categories = list(set(app['category'] for app in appliances))
        
        return {
            'total_appliances': len(appliances),
            'total_daily_consumption': total_daily,
            'peak_surge_power': peak_surge,
            'categories': categories
        }

class MockLocationIntelligenceAgent:
    """Mock LocationIntelligenceAgent for testing"""
    
    def __init__(self):
        self.initialized = True
        self.test_data = {
            (6.4281, 3.4219): {  # Lagos coordinates
                'annual_average': 5.2,  # kWh/m²/day
                'monthly_values': [5.1, 5.3, 5.4, 5.2, 4.9, 4.7, 4.5, 4.6, 4.8, 5.0, 5.2, 5.3],
                'cloud_cover': 'Moderate',
                'rainfall_pattern': 'High during rainy season',
                'seasonal_variations': 'Distinct wet and dry seasons'
            }
        }
    
    def validate_location_data(self, location_data: Dict[str, Any]) -> bool:
        """Validate location data structure"""
        required_fields = ['state', 'city', 'lga', 'full_address', 'coordinates']
        if not all(field in location_data for field in required_fields):
            raise ValueError(f"Missing required fields in location data: {required_fields}")
        return True
    
    def get_solar_irradiance(self, coordinates: tuple) -> Dict[str, Any]:
        """Get solar irradiance data for location"""
        if coordinates in self.test_data:
            data = self.test_data[coordinates]
            return {
                'annual_average': data['annual_average'],
                'monthly_values': data['monthly_values']
            }
        return {
            'annual_average': 5.0,
            'monthly_values': [5.0] * 12
        }
    
    def get_weather_patterns(self, coordinates: tuple) -> Dict[str, Any]:
        """Get weather pattern data for location"""
        if coordinates in self.test_data:
            data = self.test_data[coordinates]
            return {
                'cloud_cover': data['cloud_cover'],
                'rainfall_pattern': data['rainfall_pattern'],
                'seasonal_variations': data['seasonal_variations']
            }
        return {
            'cloud_cover': 'Unknown',
            'rainfall_pattern': 'Unknown',
            'seasonal_variations': 'Unknown'
        }
    
    def calculate_optimal_panel_angle(self, coordinates: tuple) -> Dict[str, Any]:
        """Calculate optimal solar panel mounting angles"""
        latitude = coordinates[0]
        return {
            'tilt_angle': abs(latitude) * 0.87,  # General rule of thumb
            'azimuth_angle': 180 if latitude < 0 else 0  # South in Northern hemisphere, North in Southern
        }
    
    def generate_location_report(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive location report"""
        coordinates = location_data['coordinates']
        solar_data = self.get_solar_irradiance(coordinates)
        weather_data = self.get_weather_patterns(coordinates)
        angle_data = self.calculate_optimal_panel_angle(coordinates)
        
        return {
            'location_summary': {
                'state': location_data['state'],
                'city': location_data['city'],
                'coordinates': coordinates
            },
            'solar_potential': {
                'solar_irradiance': solar_data['annual_average'],
                'monthly_variations': solar_data['monthly_values']
            },
            'weather_impact': weather_data,
            'installation_recommendations': {
                'optimal_angles': angle_data,
                'notes': 'Based on location-specific calculations'
            }
        }

class MockEnhancedSystemSizingAgent:
    """Mock EnhancedSystemSizingAgent for testing"""
    
    def __init__(self):
        self.initialized = True
    
    def calculate_battery_capacity(self, daily_consumption: float, autonomy_days: int = 2) -> Dict[str, Any]:
        """Calculate required battery capacity with DoD and efficiency factors"""
        raw_capacity = daily_consumption * autonomy_days
        actual_capacity = raw_capacity * 1.2  # Adding 20% for depth of discharge
        
        return {
            'capacity_wh': actual_capacity,
            'recommended_voltage': 48 if actual_capacity > 5000 else 24,
            'amp_hours': actual_capacity / 48 if actual_capacity > 5000 else actual_capacity / 24
        }
    
    def calculate_solar_array_size(self, daily_energy_need: float, location_coordinates: tuple) -> Dict[str, Any]:
        """Calculate required solar array size based on location and energy needs"""
        # Assume 5 sun hours per day and system efficiency of 0.8
        sun_hours = 5
        system_efficiency = 0.8
        
        total_watts_peak = (daily_energy_need / sun_hours) / system_efficiency
        panel_wattage = 400  # Standard panel size
        
        return {
            'total_watts_peak': total_watts_peak,
            'recommended_panel_count': round(total_watts_peak / panel_wattage),
            'panel_wattage': panel_wattage
        }
    
    def size_inverter(self, continuous_power: float, surge_power: float) -> Dict[str, Any]:
        """Size inverter based on power requirements"""
        inverter_continuous = continuous_power * 1.2  # 20% safety margin
        inverter_surge = max(surge_power * 1.5, inverter_continuous * 2)
        
        return {
            'capacity_watts': inverter_continuous,
            'surge_capacity': inverter_surge,
            'recommended_type': 'pure_sine_wave'
        }
    
    def optimize_system_cost(self, system_specs: Dict[str, Any], budget: float) -> Dict[str, Any]:
        """Optimize system specifications within budget constraints"""
        original_cost = (
            system_specs['battery']['capacity_wh'] * 0.2 +  # Battery cost estimate
            system_specs['solar_array']['total_watts_peak'] * 0.8 +  # Solar array cost estimate
            system_specs['inverter']['capacity_watts'] * 0.3  # Inverter cost estimate
        )
        
        if original_cost <= budget:
            return {
                'original_cost': original_cost,
                'optimized_cost': original_cost,
                'cost_savings': 0,
                'modifications': []
            }
        
        # Simple cost reduction strategy
        optimized_specs = system_specs.copy()
        cost_reduction = 0.2  # 20% reduction
        
        return {
            'original_cost': original_cost,
            'optimized_cost': original_cost * (1 - cost_reduction),
            'cost_savings': original_cost * cost_reduction,
            'modifications': ['Reduced battery capacity', 'Adjusted solar array size']
        }
    
    def generate_system_report(self, appliance_data: Dict[str, Any], 
                             location_data: Dict[str, Any],
                             system_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive system sizing report"""
        daily_consumption = 4200  # Example from sample appliance data
        autonomy_days = system_preferences['autonomy_days']
        
        battery_specs = self.calculate_battery_capacity(daily_consumption, autonomy_days)
        solar_specs = self.calculate_solar_array_size(daily_consumption, location_data['coordinates'])
        inverter_specs = self.size_inverter(250, 570)  # From sample data
        
        system_specs = {
            'battery': battery_specs,
            'solar_array': solar_specs,
            'inverter': inverter_specs
        }
        
        optimized = self.optimize_system_cost(system_specs, system_preferences['budget_amount'])
        
        return {
            'system_specifications': system_specs,
            'cost_analysis': {
                'total_cost': optimized['optimized_cost'],
                'savings': optimized['cost_savings'],
                'modifications': optimized['modifications']
            },
            'installation_requirements': {
                'panel_area_needed': solar_specs['recommended_panel_count'] * 2,  # 2m² per panel
                'battery_bank_specs': battery_specs
            },
            'maintenance_recommendations': {
                'battery_maintenance': 'Quarterly checks recommended',
                'panel_cleaning': 'Monthly cleaning recommended',
                'inverter_inspection': 'Annual professional inspection'
            }
        }

class MockBrandIntelligenceAgent:
    """Mock BrandIntelligenceAgent for testing"""
    
    def __init__(self):
        self.initialized = True