"""Unit tests for LocationIntelligenceAgent"""
import pytest
from tests.unit.test_inputMapping_agents import MockLocationIntelligenceAgent

def test_location_intelligence_agent_initialization(location_intelligence_agent):
    """Test that LocationIntelligenceAgent initializes correctly"""
    assert isinstance(location_intelligence_agent, MockLocationIntelligenceAgent)
    assert location_intelligence_agent.initialized == True
    assert isinstance(location_intelligence_agent.test_data, dict)
    assert len(location_intelligence_agent.test_data) > 0

def test_validate_location_data(location_intelligence_agent, sample_location_data):
    """Test location data validation"""
    # Test with valid data
    assert location_intelligence_agent.validate_location_data(sample_location_data) is True
    
    # Test with invalid data
    invalid_data = {
        'state': 'Invalid State',
        'city': 'Invalid City'
        # Missing required fields
    }
    with pytest.raises(ValueError):
        location_intelligence_agent.validate_location_data(invalid_data)

def test_get_solar_irradiance(location_intelligence_agent, sample_location_data):
    """Test retrieval of solar irradiance data"""
    irradiance_data = location_intelligence_agent.get_solar_irradiance(
        sample_location_data['coordinates']
    )
    
    assert isinstance(irradiance_data, dict)
    assert 'annual_average' in irradiance_data
    assert 'monthly_values' in irradiance_data
    assert len(irradiance_data['monthly_values']) == 12
    assert all(isinstance(v, (int, float)) for v in irradiance_data['monthly_values'])

def test_get_weather_patterns(location_intelligence_agent, sample_location_data):
    """Test retrieval of weather pattern data"""
    weather_data = location_intelligence_agent.get_weather_patterns(
        sample_location_data['coordinates']
    )
    
    assert isinstance(weather_data, dict)
    assert 'rainfall_pattern' in weather_data
    assert 'cloud_cover' in weather_data
    assert 'seasonal_variations' in weather_data

def test_calculate_optimal_panel_angle(location_intelligence_agent, sample_location_data):
    """Test calculation of optimal solar panel angle"""
    angle_data = location_intelligence_agent.calculate_optimal_panel_angle(
        sample_location_data['coordinates']
    )
    
    assert isinstance(angle_data, dict)
    assert 'tilt_angle' in angle_data
    assert 'azimuth_angle' in angle_data
    assert isinstance(angle_data['tilt_angle'], (int, float))
    assert isinstance(angle_data['azimuth_angle'], (int, float))

def test_generate_location_report(location_intelligence_agent, sample_location_data):
    """Test generation of comprehensive location report"""
    report = location_intelligence_agent.generate_location_report(sample_location_data)
    
    assert isinstance(report, dict)
    assert 'location_summary' in report
    assert 'solar_potential' in report
    assert 'weather_impact' in report
    assert 'installation_recommendations' in report
    
    # Verify report content
    assert report['location_summary']['state'] == sample_location_data['state']
    assert report['location_summary']['city'] == sample_location_data['city']
    assert 'solar_irradiance' in report['solar_potential']
    assert 'optimal_angles' in report['installation_recommendations']