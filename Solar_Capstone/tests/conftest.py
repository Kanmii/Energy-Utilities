"""Pytest configuration for agent testing"""
import pytest
import os
import sys
from typing import Dict, Any

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import agents and utilities
from backend.app.agents.input_mapping_agent import InputMappingAgent
from backend.app.agents.location_intelligence_agent import LocationIntelligenceAgent
from backend.app.agents.system_sizing_agent import EnhancedSystemSizingAgent
from backend.app.agents.brand_intelligence_agent import BrandIntelligenceAgent
from backend.app.agents.chat_interface_agent import ChatInterfaceAgent
from backend.app.agents.super_agent import SuperAgent

@pytest.fixture
def sample_appliance_data() -> Dict[str, Any]:
    """Sample appliance data for testing"""
    return {
        'appliances': [
            {
                'name': 'LED TV - 43 inch',
                'category': 'Electronics',
                'type': '43 inch LED TV',
                'usage_hours': 6,
                'quantity': 1,
                'power_watts': 100,
                'min_power_w': 80,
                'max_power_w': 120,
                'surge_factor': 1.2
            },
            {
                'name': 'Refrigerator - Medium Size',
                'category': 'Appliances',
                'type': 'Double Door',
                'usage_hours': 24,
                'quantity': 1,
                'power_watts': 150,
                'min_power_w': 120,
                'max_power_w': 180,
                'surge_factor': 3.0
            }
        ]
    }

@pytest.fixture
def sample_location_data() -> Dict[str, Any]:
    """Sample location data for testing"""
    return {
        'state': 'Lagos',
        'city': 'Victoria Island',
        'lga': 'Eti-Osa',
        'full_address': '123 Main Street, Victoria Island, Lagos',
        'coordinates': (6.4281, 3.4219)
    }

@pytest.fixture
def sample_system_preferences() -> Dict[str, Any]:
    """Sample system preferences for testing"""
    return {
        'autonomy_days': 2,
        'budget_amount': 2000000,
        'budget_range': 'medium'
    }

@pytest.fixture
def input_mapping_agent():
    """Input mapping agent fixture"""
    return InputMappingAgent()

@pytest.fixture
def location_intelligence_agent():
    """Location intelligence agent fixture"""
    return LocationIntelligenceAgent()

@pytest.fixture
def system_sizing_agent():
    """System sizing agent fixture"""
    return EnhancedSystemSizingAgent()

@pytest.fixture
def brand_intelligence_agent():
    """Brand intelligence agent fixture"""
    return BrandIntelligenceAgent()

@pytest.fixture
def chat_interface_agent():
    """Chat interface agent fixture"""
    return ChatInterfaceAgent()

@pytest.fixture
def super_agent():
    """Super agent fixture"""
    return SuperAgent()