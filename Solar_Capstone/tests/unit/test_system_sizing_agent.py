"""Unit tests for EnhancedSystemSizingAgent"""
import pytest
from backend.app.agents.system_sizing_agent import EnhancedSystemSizingAgent

def test_system_sizing_agent_initialization(system_sizing_agent):
    """Test that EnhancedSystemSizingAgent initializes correctly"""
    assert isinstance(system_sizing_agent, EnhancedSystemSizingAgent)

def test_calculate_battery_capacity(system_sizing_agent, sample_appliance_data, sample_system_preferences):
    """Test battery capacity calculation"""
    daily_consumption = 4200  # From sample data (TV + Fridge)
    battery_capacity = system_sizing_agent.calculate_battery_capacity(
        daily_consumption,
        autonomy_days=sample_system_preferences['autonomy_days']
    )
    
    assert isinstance(battery_capacity, dict)
    assert 'capacity_wh' in battery_capacity
    assert 'recommended_voltage' in battery_capacity
    assert 'amp_hours' in battery_capacity
    
    # Verify calculations
    # Expected capacity for 2 days autonomy with 4200Wh daily consumption
    expected_capacity = 4200 * 2 * 1.2  # Including depth of discharge factor
    assert battery_capacity['capacity_wh'] == expected_capacity

def test_calculate_solar_array_size(system_sizing_agent, sample_location_data):
    """Test solar array sizing calculation"""
    daily_energy_need = 4200  # From sample data
    array_size = system_sizing_agent.calculate_solar_array_size(
        daily_energy_need,
        sample_location_data['coordinates']
    )
    
    assert isinstance(array_size, dict)
    assert 'total_watts_peak' in array_size
    assert 'recommended_panel_count' in array_size
    assert 'panel_wattage' in array_size
    assert array_size['total_watts_peak'] > 0
    assert array_size['recommended_panel_count'] > 0

def test_size_inverter(system_sizing_agent, sample_appliance_data):
    """Test inverter sizing calculation"""
    surge_power = 570  # From sample data
    continuous_power = 250  # TV + Fridge running power
    
    inverter_spec = system_sizing_agent.size_inverter(continuous_power, surge_power)
    
    assert isinstance(inverter_spec, dict)
    assert 'capacity_watts' in inverter_spec
    assert 'surge_capacity' in inverter_spec
    assert 'recommended_type' in inverter_spec
    assert inverter_spec['capacity_watts'] >= continuous_power
    assert inverter_spec['surge_capacity'] >= surge_power

def test_optimize_system_cost(system_sizing_agent, sample_system_preferences):
    """Test system cost optimization"""
    system_specs = {
        'battery': {'capacity_wh': 10080},  # From previous calculation
        'solar_array': {'total_watts_peak': 2000},
        'inverter': {'capacity_watts': 1000}
    }
    
    optimized_specs = system_sizing_agent.optimize_system_cost(
        system_specs,
        sample_system_preferences['budget_amount']
    )
    
    assert isinstance(optimized_specs, dict)
    assert 'original_cost' in optimized_specs
    assert 'optimized_cost' in optimized_specs
    assert 'cost_savings' in optimized_specs
    assert 'modifications' in optimized_specs
    assert optimized_specs['optimized_cost'] <= sample_system_preferences['budget_amount']

def test_generate_system_report(system_sizing_agent, sample_appliance_data, 
                              sample_location_data, sample_system_preferences):
    """Test generation of comprehensive system sizing report"""
    report = system_sizing_agent.generate_system_report(
        sample_appliance_data,
        sample_location_data,
        sample_system_preferences
    )
    
    assert isinstance(report, dict)
    assert 'system_specifications' in report
    assert 'cost_analysis' in report
    assert 'installation_requirements' in report
    assert 'maintenance_recommendations' in report
    
    # Verify report content
    specs = report['system_specifications']
    assert 'solar_array' in specs
    assert 'battery_bank' in specs
    assert 'inverter' in specs
    assert 'total_cost' in report['cost_analysis']