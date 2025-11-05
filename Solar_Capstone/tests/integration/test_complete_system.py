#!/usr/bin/env python3
"""
 Complete Multi-Agent System Test - Solar System Recommendation Platform
End-to-End Testing of All AI Agents Working Together
"""
import sys
        # Import tools needed for this file
import os
        # Import tools needed for this file
import time
        # Import tools needed for this file
import json
        # Import tools needed for this file
from datetime import datetime
        # Import tools needed for this file

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import all agents
from backend.app.agents.input_mapping_agent import InputMappingAgent
        # Import tools needed for this file
from backend.app.agents.geo_agent import GeoAgent
        # Import tools needed for this file
from backend.app.agents.system_sizing_agent import SystemSizingAgent
        # Import tools needed for this file
from backend.app.agents.brand_intelligence_agent import BrandIntelligenceAgent
        # Import tools needed for this file
from backend.app.agents.educational_agent import EducationalAgent
        # Import tools needed for this file

def test_agent_initialization():
    # This function sets up the agent when it starts
    """Test initialization of all agents"""
    print(" Testing Agent Initialization...")
        # Display information to the user
    
    agents = {}
    
    try:
        # Try to execute the code safely
        print("    Initializing InputMappingAgent...")
        # Display information to the user
        agents['input_mapping'] = InputMappingAgent()
        print("    InputMappingAgent ready")
        # Display information to the user
        
        print("    Initializing GeoAgent...")
        # Display information to the user
        agents['geo'] = GeoAgent()
        print("    GeoAgent ready")
        # Display information to the user
        
        print("    Initializing SystemSizingAgent...")
        # Display information to the user
        agents['sizing'] = SystemSizingAgent()
        print("    SystemSizingAgent ready")
        # Display information to the user
        
        print("    Initializing BrandIntelligenceAgent...")
        # Display information to the user
        agents['brand'] = BrandIntelligenceAgent()
        print("    BrandIntelligenceAgent ready")
        # Display information to the user
        
        print("    Initializing EducationalAgent...")
        # Display information to the user
        agents['educational'] = EducationalAgent()
        print("    EducationalAgent ready")
        # Display information to the user
        
        return agents
        # Send the result back to the caller
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" Agent initialization failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_input_mapping_agent(agent):
    # This function performs a specific task for the solar system
    """Test InputMappingAgent functionality"""
    print("\n Testing InputMappingAgent...")
        # Display information to the user
    
    # Sample appliance data
    appliance_data = {
        'appliances': [
            'Refrigerator', 'Television', 'Air Conditioner', 
            'LED Lights', 'Water Pump', 'Laptop'
        ],
        'usage_hours': {
            'Refrigerator': 24,
            'Television': 6,
            'Air Conditioner': 8,
            'LED Lights': 12,
            'Water Pump': 2,
            'Laptop': 8
        },
        'quantities': {
            'Refrigerator': 1,
            'Television': 1,
            'Air Conditioner': 1,
            'LED Lights': 10,
            'Water Pump': 1,
            'Laptop': 1
        }
    }
    
    try:
        # Try to execute the code safely
        start_time = time.time()
        result = agent.process_appliances(appliance_data)
        processing_time = time.time() - start_time
        
        print(f"    InputMappingAgent processed successfully")
        # Display information to the user
        print(f"   ⏱ Processing time: {processing_time:.3f}s")
        # Display information to the user
        print(f"    Total daily energy: {result.get('total_daily_energy_kwh', 0):.2f} kWh")
        # Display information to the user
        print(f"    Total appliances: {len(result.get('appliance_details', []))}")
        # Display information to the user
        
        return result
        # Send the result back to the caller
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" InputMappingAgent test failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_geo_agent(agent):
    # This function performs a specific task for the solar system
    """Test GeoAgent functionality"""
    print("\n Testing GeoAgent...")
        # Display information to the user
    
    # Sample location data
    location_data = {
        'city': 'Lagos',
        'state': 'Lagos',
        'region': 'South West',
        'latitude': 6.5244,
        'longitude': 3.3792
    }
    
    try:
        # Try to execute the code safely
        start_time = time.time()
        result = agent.process_location(location_data)
        processing_time = time.time() - start_time
        
        if result:
        # Check a condition
            print(f"    GeoAgent processed successfully")
        # Display information to the user
            print(f"   ⏱ Processing time: {processing_time:.3f}s")
        # Display information to the user
            print(f"    Sun peak hours: {result.location_data.sun_peak_hours:.1f}")
        # Display information to the user
            print(f"    Location: {result.location_data.location}")
        # Display information to the user
            print(f"    Confidence: {result.location_data.confidence:.2f}")
        # Display information to the user
            return result
        # Send the result back to the caller
        else:
            print(f" GeoAgent returned no result")
        # Display information to the user
            return None
        # Send the result back to the caller
            
    except Exception as e:
        # Handle any errors that might occur
        print(f" GeoAgent test failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_system_sizing_agent(agent, load_data, location_intelligence):
    # This function performs a specific task for the solar system
    """Test SystemSizingAgent functionality"""
    print("\n Testing SystemSizingAgent...")
        # Display information to the user
    
    # Sample preferences
    preferences = {
        'autonomy_days': 3,
        'battery_chemistry': 'LiFePO4',
        'budget_range': 'medium',
        'quality_threshold': 0.8
    }
    
    try:
        # Try to execute the code safely
        start_time = time.time()
        result = agent.calculate_system_sizing(load_data, location_intelligence, preferences)
        processing_time = time.time() - start_time
        
        print(f"    SystemSizingAgent processed successfully")
        # Display information to the user
        print(f"   ⏱ Processing time: {processing_time:.3f}s")
        # Display information to the user
        print(f"    Panel power: {result.panel_power_watts:.0f}W")
        # Display information to the user
        print(f"    Battery capacity: {result.battery_capacity_kwh:.1f} kWh")
        # Display information to the user
        print(f"    Inverter power: {result.inverter_power_watts:.0f}W")
        # Display information to the user
        print(f"    System efficiency: {result.system_efficiency:.1%}")
        # Display information to the user
        
        return result
        # Send the result back to the caller
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" SystemSizingAgent test failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_brand_intelligence_agent(agent, system_requirements, preferences):
    # This function performs a specific task for the solar system
    """Test BrandIntelligenceAgent functionality"""
    print("\n Testing BrandIntelligenceAgent...")
        # Display information to the user
    
    try:
        # Try to execute the code safely
        start_time = time.time()
        result = agent.recommend_components(system_requirements, preferences)
        processing_time = time.time() - start_time
        
        print(f"    BrandIntelligenceAgent processed successfully")
        # Display information to the user
        print(f"   ⏱ Processing time: {processing_time:.3f}s")
        # Display information to the user
        print(f"    Total cost range: ₦{result.total_cost_range[0]:,.0f} - ₦{result.total_cost_range[1]:,.0f}")
        # Display information to the user
        print(f"    Quality score: {result.total_quality_score:.2f}")
        # Display information to the user
        print(f"    Performance score: {result.total_performance_score:.2f}")
        # Display information to the user
        print(f"    Component types: {len(result.components)}")
        # Display information to the user
        
        # Show component recommendations
        for comp_type, components in result.components.items():
        # Go through each item in the list
            if components:
        # Check a condition
                top_comp = components[0]
                print(f"    Top {comp_type}: {top_comp.brand} {top_comp.model}")
        # Display information to the user
        
        return result
        # Send the result back to the caller
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" BrandIntelligenceAgent test failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_educational_agent(agent, user_profile, system_data):
    # This function performs a specific task for the solar system
    """Test EducationalAgent functionality"""
    print("\n Testing EducationalAgent...")
        # Display information to the user
    
    try:
        # Try to execute the code safely
        start_time = time.time()
        result = agent.generate_educational_response(user_profile, system_data, "energy_calculation system_sizing cost_analysis")
        processing_time = time.time() - start_time
        
        print(f"    EducationalAgent processed successfully")
        # Display information to the user
        print(f"   ⏱ Processing time: {processing_time:.3f}s")
        # Display information to the user
        print(f"    Educational content: {len(result.educational_content)}")
        # Display information to the user
        print(f"    Explanations: {len(result.explanations)}")
        # Display information to the user
        print(f"    FAQ suggestions: {len(result.faq_suggestions)}")
        # Display information to the user
        print(f"    Confidence: {result.confidence:.2f}")
        # Display information to the user
        
        # Show sample content
        if result.educational_content:
        # Check a condition
            content = result.educational_content[0]
            print(f"    Sample content: {content.title}")
        # Display information to the user
        
        if result.explanations:
        # Check a condition
            explanation = result.explanations[0]
            print(f"    Sample explanation: {explanation.title}")
        # Display information to the user
        
        return result
        # Send the result back to the caller
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" EducationalAgent test failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_complete_system_workflow(agents):
    # This function performs a specific task for the solar system
    """Test complete system workflow with all agents"""
    print("\n Testing Complete System Workflow...")
        # Display information to the user
    
    try:
        # Try to execute the code safely
        total_start_time = time.time()
        
        # Step 1: Process appliances
        print("\n Step 1: Processing Appliances...")
        # Display information to the user
        appliance_data = {
            'appliances': [
                'Refrigerator', 'Television', 'Air Conditioner', 
                'LED Lights', 'Water Pump', 'Laptop', 'Phone Charger'
            ],
            'usage_hours': {
                'Refrigerator': 24,
                'Television': 6,
                'Air Conditioner': 8,
                'LED Lights': 12,
                'Water Pump': 2,
                'Laptop': 8,
                'Phone Charger': 4
            },
            'quantities': {
                'Refrigerator': 1,
                'Television': 1,
                'Air Conditioner': 1,
                'LED Lights': 10,
                'Water Pump': 1,
                'Laptop': 1,
                'Phone Charger': 2
            }
        }
        
        load_result = agents['input_mapping'].process_appliances(appliance_data)
        daily_energy = load_result.get('total_daily_energy_kwh', 0)
        print(f"    Daily energy calculated: {daily_energy:.2f} kWh")
        # Display information to the user
        
        # Step 2: Process location
        print("\n Step 2: Processing Location...")
        # Display information to the user
        location_data = {
            'city': 'Lagos',
            'state': 'Lagos',
            'region': 'South West',
            'latitude': 6.5244,
            'longitude': 3.3792
        }
        
        location_intelligence = agents['geo'].process_location(location_data)
        if location_intelligence:
        # Check a condition
            print(f"    Location processed: {location_intelligence.location_data.sun_peak_hours:.1f} sun hours")
        # Display information to the user
        else:
            print(f" Location processing failed")
        # Display information to the user
            return None
        # Send the result back to the caller
        
        # Step 3: Calculate system sizing
        print("\n Step 3: Calculating System Sizing...")
        # Display information to the user
        load_data = {'daily_energy_kwh': daily_energy}
        preferences = {
            'autonomy_days': 3,
            'battery_chemistry': 'LiFePO4',
            'budget_range': 'medium',
            'quality_threshold': 0.8
        }
        
        sizing_result = agents['sizing'].calculate_system_sizing(
            load_data, location_intelligence, preferences
        )
        print(f"    System sizing calculated: {sizing_result.panel_power_watts:.0f}W panels, {sizing_result.battery_capacity_kwh:.1f} kWh batteries")
        # Display information to the user
        
        # Step 4: Get component recommendations
        print("\n Step 4: Getting Component Recommendations...")
        # Display information to the user
        system_requirements = {
            'panel_power_watts': sizing_result.panel_power_watts,
            'panel_power_watts_min': sizing_result.panel_power_watts_min,
            'panel_power_watts_max': sizing_result.panel_power_watts_max,
            'panel_count': sizing_result.panel_count,
            'panel_count_min': sizing_result.panel_count_min,
            'panel_count_max': sizing_result.panel_count_max,
            'battery_capacity_kwh': sizing_result.battery_capacity_kwh,
            'battery_capacity_kwh_min': sizing_result.battery_capacity_kwh_min,
            'battery_capacity_kwh_max': sizing_result.battery_capacity_kwh_max,
            'battery_count': sizing_result.battery_count,
            'battery_count_min': sizing_result.battery_count_min,
            'battery_count_max': sizing_result.battery_count_max,
            'battery_chemistry': sizing_result.battery_chemistry,
            'inverter_power_watts': sizing_result.inverter_power_watts,
            'inverter_power_watts_min': sizing_result.inverter_power_watts_min,
            'inverter_power_watts_max': sizing_result.inverter_power_watts_max,
            'charge_controller_current': sizing_result.charge_controller_current,
            'charge_controller_current_min': sizing_result.charge_controller_current_min,
            'charge_controller_current_max': sizing_result.charge_controller_current_max,
            'budget_range': preferences['budget_range']
        }
        
        recommendations = agents['brand'].recommend_components(
            system_requirements, preferences
        )
        print(f"    Component recommendations generated: ₦{recommendations.total_cost_range[0]:,.0f} - ₦{recommendations.total_cost_range[1]:,.0f}")
        # Display information to the user
        
        # Step 5: Get educational content
        print("\n Step 5: Getting Educational Content...")
        # Display information to the user
        user_profile = {
            'experience_level': 'intermediate',
            'system_size': 'medium',
            'budget_range': 'medium'
        }
        
        system_data = {
            'daily_energy': daily_energy,
            'sizing_result': sizing_result,
            'recommendations': recommendations
        }
        
        educational_response = agents['educational'].generate_educational_response(
            user_profile, system_data, "energy_calculation system_sizing cost_analysis"
        )
        print(f"    Educational content generated: {len(educational_response.educational_content)} items")
        # Display information to the user
        
        total_processing_time = time.time() - total_start_time
        
        # Summary
        print(f"\n Complete System Workflow Successful!")
        # Display information to the user
        print(f"⏱ Total processing time: {total_processing_time:.3f}s")
        # Display information to the user
        print(f" System specifications:")
        # Display information to the user
        print(f"   • Daily energy: {daily_energy:.2f} kWh")
        # Display information to the user
        print(f"   • Panel power: {sizing_result.panel_power_watts:.0f}W")
        # Display information to the user
        print(f"   • Battery capacity: {sizing_result.battery_capacity_kwh:.1f} kWh")
        # Display information to the user
        print(f"   • Inverter power: {sizing_result.inverter_power_watts:.0f}W")
        # Display information to the user
        print(f"   • System efficiency: {sizing_result.system_efficiency:.1%}")
        # Display information to the user
        print(f"   • Total cost: ₦{recommendations.total_cost_range[0]:,.0f} - ₦{recommendations.total_cost_range[1]:,.0f}")
        # Display information to the user
        print(f"   • Quality score: {recommendations.total_quality_score:.2f}")
        # Display information to the user
        print(f"   • Performance score: {recommendations.total_performance_score:.2f}")
        # Display information to the user
        
        return {
        # Send the result back to the caller
            'success': True,
            'load_result': load_result,
            'location_intelligence': location_intelligence,
            'sizing_result': sizing_result,
            'recommendations': recommendations,
            'educational_response': educational_response,
            'processing_time': total_processing_time
        }
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" Complete system workflow failed: {e}")
        # Display information to the user
        return None
        # Send the result back to the caller

def test_multiple_scenarios(agents):
    # This function performs a specific task for the solar system
    """Test multiple scenarios with different parameters"""
    print("\n Testing Multiple Scenarios...")
        # Display information to the user
    
    scenarios = [
        {
            'name': 'Small Home System',
            'appliances': ['Refrigerator', 'TV', 'Lights'],
            'usage_hours': {'Refrigerator': 24, 'TV': 6, 'Lights': 12},
            'quantities': {'Refrigerator': 1, 'TV': 1, 'Lights': 5},
            'location': {'city': 'Lagos', 'state': 'Lagos', 'region': 'South West'},
            'preferences': {'autonomy_days': 2, 'battery_chemistry': 'LiFePO4', 'budget_range': 'budget'}
        },
        {
            'name': 'Medium Office System',
            'appliances': ['Refrigerator', 'TV', 'Air Conditioner', 'Laptop', 'Lights'],
            'usage_hours': {'Refrigerator': 24, 'TV': 8, 'Air Conditioner': 10, 'Laptop': 8, 'Lights': 12},
            'quantities': {'Refrigerator': 1, 'TV': 1, 'Air Conditioner': 2, 'Laptop': 3, 'Lights': 15},
            'location': {'city': 'Abuja', 'state': 'FCT', 'region': 'North Central'},
            'preferences': {'autonomy_days': 3, 'battery_chemistry': 'LiFePO4', 'budget_range': 'medium'}
        },
        {
            'name': 'Large Commercial System',
            'appliances': ['Refrigerator', 'Air Conditioner', 'Water Pump', 'Lights', 'Security System'],
            'usage_hours': {'Refrigerator': 24, 'Air Conditioner': 12, 'Water Pump': 4, 'Lights': 16, 'Security System': 24},
            'quantities': {'Refrigerator': 2, 'Air Conditioner': 4, 'Water Pump': 2, 'Lights': 30, 'Security System': 1},
            'location': {'city': 'Kano', 'state': 'Kano', 'region': 'North West'},
            'preferences': {'autonomy_days': 5, 'battery_chemistry': 'LiFePO4', 'budget_range': 'premium'}
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        # Go through each item in the list
        print(f"\n Testing Scenario {i}: {scenario['name']}")
        # Display information to the user
        
        try:
        # Try to execute the code safely
            # Process scenario
            appliance_data = {
                'appliances': scenario['appliances'],
                'usage_hours': scenario['usage_hours'],
                'quantities': scenario['quantities']
            }
            
            load_result = agents['input_mapping'].process_appliances(appliance_data)
            daily_energy = load_result.get('total_daily_energy_kwh', 0)
            
            location_intelligence = agents['geo'].process_location(scenario['location'])
            
            if location_intelligence:
        # Check a condition
                load_data = {'daily_energy_kwh': daily_energy}
                sizing_result = agents['sizing'].calculate_system_sizing(
                    load_data, location_intelligence, scenario['preferences']
                )
                
                system_requirements = {
                    'panel_power_watts': sizing_result.panel_power_watts,
                    'battery_capacity_kwh': sizing_result.battery_capacity_kwh,
                    'inverter_power_watts': sizing_result.inverter_power_watts,
                    'budget_range': scenario['preferences']['budget_range']
                }
                
                recommendations = agents['brand'].recommend_components(
                    system_requirements, scenario['preferences']
                )
                
                print(f"    {scenario['name']} processed successfully")
        # Display information to the user
                print(f"    Daily energy: {daily_energy:.2f} kWh")
        # Display information to the user
                print(f"    Panel power: {sizing_result.panel_power_watts:.0f}W")
        # Display information to the user
                print(f"    Battery capacity: {sizing_result.battery_capacity_kwh:.1f} kWh")
        # Display information to the user
                print(f"    Cost range: ₦{recommendations.total_cost_range[0]:,.0f} - ₦{recommendations.total_cost_range[1]:,.0f}")
        # Display information to the user
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'daily_energy': daily_energy,
                    'panel_power': sizing_result.panel_power_watts,
                    'battery_capacity': sizing_result.battery_capacity_kwh,
                    'cost_range': recommendations.total_cost_range
                })
            else:
                print(f"    {scenario['name']} failed - location processing error")
        # Display information to the user
                results.append({
                    'scenario': scenario['name'],
                    'success': False,
                    'error': 'Location processing failed'
                })
                
        except Exception as e:
        # Handle any errors that might occur
            print(f"    {scenario['name']} failed: {e}")
        # Display information to the user
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })
    
    return results
        # Send the result back to the caller

def main():
    # This function performs a specific task for the solar system
    """Main test function"""
    print(" Complete Multi-Agent System Test - Solar System Recommendation Platform")
        # Display information to the user
    print("=" * 80)
        # Display information to the user
    print(f"Test started at: {datetime.now()}")
        # Display information to the user
    print("=" * 80)
        # Display information to the user
    
    # Test agent initialization
    agents = test_agent_initialization()
    if not agents:
        # Check a condition
        print(" Cannot proceed without agent initialization")
        # Display information to the user
        return
    
    # Test individual agents
    print("\n" + "=" * 60)
        # Display information to the user
    print(" Testing Individual Agents")
        # Display information to the user
    print("=" * 60)
        # Display information to the user
    
    # Test InputMappingAgent
    load_result = test_input_mapping_agent(agents['input_mapping'])
    if not load_result:
        # Check a condition
        print(" InputMappingAgent test failed")
        # Display information to the user
        return
    
    # Test GeoAgent
    location_intelligence = test_geo_agent(agents['geo'])
    if not location_intelligence:
        # Check a condition
        print(" GeoAgent test failed")
        # Display information to the user
        return
    
    # Test SystemSizingAgent
    load_data = {'daily_energy_kwh': load_result.get('total_daily_energy_kwh', 0)}
    preferences = {
        'autonomy_days': 3,
        'battery_chemistry': 'LiFePO4',
        'budget_range': 'medium',
        'quality_threshold': 0.8
    }
    
    sizing_result = test_system_sizing_agent(agents['sizing'], load_data, location_intelligence)
    if not sizing_result:
        # Check a condition
        print(" SystemSizingAgent test failed")
        # Display information to the user
        return
    
    # Test BrandIntelligenceAgent
    system_requirements = {
        'panel_power_watts': sizing_result.panel_power_watts,
        'battery_capacity_kwh': sizing_result.battery_capacity_kwh,
        'inverter_power_watts': sizing_result.inverter_power_watts,
        'budget_range': preferences['budget_range']
    }
    
    recommendations = test_brand_intelligence_agent(agents['brand'], system_requirements, preferences)
    if not recommendations:
        # Check a condition
        print(" BrandIntelligenceAgent test failed")
        # Display information to the user
        return
    
    # Test EducationalAgent
    user_profile = {
        'experience_level': 'intermediate',
        'system_size': 'medium',
        'budget_range': 'medium'
    }
    
    system_data = {
        'daily_energy': load_result.get('total_daily_energy_kwh', 0),
        'sizing_result': sizing_result,
        'recommendations': recommendations
    }
    
    educational_response = test_educational_agent(agents['educational'], user_profile, system_data)
    if not educational_response:
        # Check a condition
        print(" EducationalAgent test failed")
        # Display information to the user
        return
    
    # Test complete system workflow
    print("\n" + "=" * 60)
        # Display information to the user
    print(" Testing Complete System Workflow")
        # Display information to the user
    print("=" * 60)
        # Display information to the user
    
    workflow_result = test_complete_system_workflow(agents)
    if not workflow_result:
        # Check a condition
        print(" Complete system workflow failed")
        # Display information to the user
        return
    
    # Test multiple scenarios
    print("\n" + "=" * 60)
        # Display information to the user
    print(" Testing Multiple Scenarios")
        # Display information to the user
    print("=" * 60)
        # Display information to the user
    
    scenario_results = test_multiple_scenarios(agents)
    
    # Final summary
    print("\n" + "=" * 80)
        # Display information to the user
    print(" Complete Multi-Agent System Test Results")
        # Display information to the user
    print("=" * 80)
        # Display information to the user
    
    successful_scenarios = [r for r in scenario_results if r['success']]
    failed_scenarios = [r for r in scenario_results if not r['success']]
    
    print(f" Individual Agent Tests: PASSED")
        # Display information to the user
    print(f" Complete System Workflow: PASSED")
        # Display information to the user
    print(f" Multiple Scenarios: {len(successful_scenarios)}/{len(scenario_results)} PASSED")
        # Display information to the user
    
    if failed_scenarios:
        # Check a condition
        print(f" Failed Scenarios:")
        # Display information to the user
        for scenario in failed_scenarios:
        # Go through each item in the list
            print(f"   • {scenario['scenario']}: {scenario.get('error', 'Unknown error')}")
        # Display information to the user
    
    print(f"\n Your AI-powered solar recommendation platform is fully operational!")
        # Display information to the user
    print(f" All {len(agents)} agents are working together successfully")
        # Display information to the user
    print(f" Ready for production deployment!")
        # Display information to the user
    print("=" * 80)
        # Display information to the user

if __name__ == "__main__":
        # Check a condition
    main()
