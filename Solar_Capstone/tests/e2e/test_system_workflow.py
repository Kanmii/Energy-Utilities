"""
End-to-end tests for the solar system sizing workflow
"""
import pytest
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# Import agents
from backend.app.agents.system_sizing_agent import SystemSizingAgent
from backend.app.agents.geo_agent import GeoAgent
from backend.app.agents.input_mapping_agent import InputMappingAgent

# Test fixtures
@pytest.fixture
def agents():
    """Initialize all required agents"""
    return {
        'input_mapping': InputMappingAgent(),
        'geo': GeoAgent(),
        'sizing': SystemSizingAgent()
    }

@pytest.fixture
def sample_user_input():
    """Sample user input data"""
    return {
        'appliances': [
            'fridge',
            'television',
            'lights',
            'fan',
            'phone charger'
        ],
        'location': {
            'city': 'Lagos',
            'state': 'Lagos',
            'region': 'South West'
        },
        'preferences': {
            'autonomy_days': 3,
            'battery_chemistry': 'LiFePO4',
            'budget_range': 'medium'
        }
    }

@pytest.fixture
def user_scenarios():
    """Different user scenarios for testing"""
    return [
        {
            'name': 'Small Apartment',
            'appliances': ['fridge', 'lights', 'fan', 'phone charger'],
            'location': {'city': 'Lagos', 'state': 'Lagos', 'region': 'South West'},
            'preferences': {'autonomy_days': 2, 'battery_chemistry': 'LiFePO4', 'budget_range': 'budget'}
        },
        {
            'name': 'Medium House',
            'appliances': ['fridge', 'television', 'lights', 'fan', 'washing machine', 'microwave'],
            'location': {'city': 'Abuja', 'state': 'FCT', 'region': 'North Central'},
            'preferences': {'autonomy_days': 3, 'battery_chemistry': 'LiFePO4', 'budget_range': 'medium'}
        },
        {
            'name': 'Large Villa',
            'appliances': ['fridge', 'television', 'lights', 'fan', 'washing machine', 'microwave', 'air conditioner', 'water pump'],
            'location': {'city': 'Port Harcourt', 'state': 'Rivers', 'region': 'South South'},
            'preferences': {'autonomy_days': 5, 'battery_chemistry': 'LiFePO4', 'budget_range': 'premium'}
        }
    ]

def test_complete_workflow(agents: Dict[str, Any], sample_user_input: Dict[str, Any]):
    """Test the complete solar system sizing workflow"""
    # Step 1: Process appliances
    appliance_input = {
        'appliances': sample_user_input['appliances'],
        'usage_hours': 'standard'
    }
    
    load_calculation = agents['input_mapping'].process_appliances(appliance_input)
    assert load_calculation is not None
    assert load_calculation['daily_energy_kwh'] > 0
    assert load_calculation['peak_power_watts'] > 0
    
    # Step 2: Process location
    location_intelligence = agents['geo'].process_location(sample_user_input['location'])
    assert location_intelligence is not None
    assert location_intelligence.location_data.sun_peak_hours > 0
    assert 0 <= location_intelligence.location_data.seasonal_variation <= 1
    assert 0 <= location_intelligence.location_data.confidence <= 1
    
    # Step 3: Calculate system sizing
    load_data = {'daily_energy_kwh': load_calculation['daily_energy_kwh']}
    sizing_result = agents['sizing'].calculate_system_sizing(
        load_data,
        location_intelligence,
        sample_user_input['preferences']
    )
    
    # Verify sizing results
    assert sizing_result is not None
    assert sizing_result.daily_energy_kwh > 0
    assert sizing_result.panel_power_watts > 0
    assert sizing_result.panel_count > 0
    assert sizing_result.battery_capacity_kwh > 0
    assert sizing_result.inverter_power_watts > 0
    assert sizing_result.estimated_cost_min > 0
    assert sizing_result.estimated_cost_max >= sizing_result.estimated_cost_min

def test_multiple_scenarios(agents: Dict[str, Any], user_scenarios: List[Dict[str, Any]]):
    """Test system sizing with different user scenarios"""
    for scenario in user_scenarios:
        # Calculate energy requirements
        energy_result = agents['input_mapping'].process_appliances({
            'appliances': scenario['appliances'],
            'usage_hours': 'standard'
        })
        assert energy_result is not None
        assert energy_result['daily_energy_kwh'] > 0
        
        # Process location
        location_intelligence = agents['geo'].process_location(scenario['location'])
        assert location_intelligence is not None
        
        # Calculate system sizing
        load_data = {'daily_energy_kwh': energy_result['daily_energy_kwh']}
        result = agents['sizing'].calculate_system_sizing(
            load_data,
            location_intelligence,
            scenario['preferences']
        )
        
        # Verify scenario-specific results
        assert result is not None
        assert result.daily_energy_kwh > 0
        assert result.panel_power_watts > 0
        assert result.panel_count > 0
        assert result.battery_capacity_kwh > 0
        assert result.inverter_power_watts > 0
        
        # Verify sizing makes sense for scenario size
        if scenario['name'] == 'Small Apartment':
            assert result.panel_count <= 8
            assert result.battery_capacity_kwh <= 10
        elif scenario['name'] == 'Large Villa':
            assert result.panel_count >= 12
            assert result.battery_capacity_kwh >= 15

def test_error_handling(agents: Dict[str, Any]):
    """Test error handling in the workflow"""
    # Test invalid location
    with pytest.raises(Exception):
        agents['geo'].process_location({
            'city': 'InvalidCity',
            'state': 'InvalidState',
            'region': 'InvalidRegion'
        })
    
    # Test invalid appliances
    with pytest.raises(Exception):
        agents['input_mapping'].process_appliances({
            'appliances': ['invalid_appliance_1', 'invalid_appliance_2'],
            'usage_hours': 'standard'
        })
    
    # Test invalid preferences
    location_intelligence = agents['geo'].process_location({
        'city': 'Lagos',
        'state': 'Lagos',
        'region': 'South West'
    })
    
    with pytest.raises(Exception):
        agents['sizing'].calculate_system_sizing(
            {'daily_energy_kwh': 10},
            location_intelligence,
            {'autonomy_days': -1, 'battery_chemistry': 'Invalid', 'budget_range': 'invalid'}
        )