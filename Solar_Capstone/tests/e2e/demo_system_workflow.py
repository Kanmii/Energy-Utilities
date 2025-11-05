#!/usr/bin/env python3
"""
Demo: Complete SystemSizingAgent Workflow
Shows how the agent works with real user input
"""
import sys
        # Import tools needed for this file
import os
        # Import tools needed for this file
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.agents.system_sizing_agent import SystemSizingAgent
        # Import tools needed for this file
from backend.app.agents.geo_agent import GeoAgent
        # Import tools needed for this file
from backend.app.agents.input_mapping_agent import InputMappingAgent
        # Import tools needed for this file

def demo_complete_workflow():
    # This function performs a specific task for the solar system
    """Demo the complete workflow from user input to system sizing"""
    print(" COMPLETE SOLAR SYSTEM WORKFLOW DEMO")
        # Display information to the user
    print("=" * 60)
        # Display information to the user
    
    # Initialize all agents
    print(" Initializing AI Agents...")
        # Display information to the user
    try:
        # Try to execute the code safely
        input_mapping_agent = InputMappingAgent()
        geo_agent = GeoAgent()
        sizing_agent = SystemSizingAgent()
        print(" All agents initialized successfully")
        # Display information to the user
    except Exception as e:
        # Handle any errors that might occur
        print(f" Error initializing agents: {e}")
        # Display information to the user
        return False
        # Send the result back to the caller
    
    # Simulate user input
    print("\n USER INPUT:")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    
    # User's appliance list
    user_appliances = [
        "fridge", "television", "lights", "fan", "phone charger"
    ]
    
    # User's location
    user_location = {
        "city": "Lagos",
        "state": "Lagos", 
        "region": "South West"
    }
    
    # User's preferences
    user_preferences = {
        "autonomy_days": 3,
        "battery_chemistry": "LiFePO4",
        "budget_range": "medium"
    }
    
    print(f"Appliances: {', '.join(user_appliances)}")
        # Display information to the user
    print(f"Location: {user_location['city']}, {user_location['state']}")
        # Display information to the user
    print(f"Autonomy: {user_preferences['autonomy_days']} days")
        # Display information to the user
    print(f"Battery: {user_preferences['battery_chemistry']}")
        # Display information to the user
    print(f"Budget: {user_preferences['budget_range']}")
        # Display information to the user
    
    # Step 1: InputMappingAgent processes appliances
    print(f"\n STEP 1: Processing Appliances with InputMappingAgent...")
        # Display information to the user
    print("-" * 50)
        # Display information to the user
    
    try:
        # Try to execute the code safely
        # Simulate appliance mapping
        appliance_input = {
            "appliances": user_appliances,
            "usage_hours": "standard"  # Use standard usage hours
        }
        
        # This would normally call: input_mapping_agent.process_appliances(appliance_input)
        # For demo, we'll simulate the result
        load_calculation = {
            "daily_energy_kwh": 12.5,  # Simulated result
            "peak_power_watts": 2500,   # Simulated result
            "appliance_details": [
                {"appliance": "fridge", "power_w": 150, "hours": 24, "energy_kwh": 3.6},
                {"appliance": "television", "power_w": 200, "hours": 6, "energy_kwh": 1.2},
                {"appliance": "lights", "power_w": 100, "hours": 8, "energy_kwh": 0.8},
                {"appliance": "fan", "power_w": 80, "hours": 12, "energy_kwh": 0.96},
                {"appliance": "phone charger", "power_w": 10, "hours": 4, "energy_kwh": 0.04}
            ]
        }
        
        print(f" Appliances processed successfully")
        # Display information to the user
        print(f"   Daily Energy: {load_calculation['daily_energy_kwh']} kWh")
        # Display information to the user
        print(f"   Peak Power: {load_calculation['peak_power_watts']}W")
        # Display information to the user
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" Error processing appliances: {e}")
        # Display information to the user
        return False
        # Send the result back to the caller
    
    # Step 2: GeoAgent processes location
    print(f"\n STEP 2: Processing Location with GeoAgent...")
        # Display information to the user
    print("-" * 50)
        # Display information to the user
    
    try:
        # Try to execute the code safely
        location_intelligence = geo_agent.process_location(user_location)
        if location_intelligence:
        # Check a condition
            print(f" Location processed successfully")
        # Display information to the user
            print(f"   Sun Peak Hours: {location_intelligence.location_data.sun_peak_hours:.1f}")
        # Display information to the user
            print(f"   Seasonal Variation: {location_intelligence.location_data.seasonal_variation:.1%}")
        # Display information to the user
            print(f"   Confidence: {location_intelligence.location_data.confidence:.1%}")
        # Display information to the user
        else:
            print(" Location processing failed")
        # Display information to the user
            return False
        # Send the result back to the caller
            
    except Exception as e:
        # Handle any errors that might occur
        print(f" Error processing location: {e}")
        # Display information to the user
        return False
        # Send the result back to the caller
    
    # Step 3: SystemSizingAgent calculates system sizing
    print(f"\n STEP 3: Calculating System Sizing...")
        # Display information to the user
    print("-" * 50)
        # Display information to the user
    
    try:
        # Try to execute the code safely
        # Prepare data for SystemSizingAgent
        load_data = {
            'daily_energy_kwh': load_calculation['daily_energy_kwh']
        }
        
        # Calculate system sizing
        sizing_result = sizing_agent.calculate_system_sizing(
            load_data, location_intelligence, user_preferences
        )
        
        print(f" System sizing calculated successfully")
        # Display information to the user
        
    except Exception as e:
        # Handle any errors that might occur
        print(f" Error calculating system sizing: {e}")
        # Display information to the user
        return False
        # Send the result back to the caller
    
    # Step 4: Display comprehensive results
    print(f"\n FINAL RESULTS:")
        # Display information to the user
    print("=" * 60)
        # Display information to the user
    
    print(f"\n ENERGY REQUIREMENTS")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print(f"Daily Energy: {sizing_result.daily_energy_kwh:.2f} kWh")
        # Display information to the user
    print(f"Range: {sizing_result.daily_energy_kwh_min:.2f} - {sizing_result.daily_energy_kwh_max:.2f} kWh")
        # Display information to the user
    
    print(f"\n SOLAR PANEL SYSTEM")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print(f"Total Power: {sizing_result.panel_power_watts:.0f}W")
        # Display information to the user
    print(f"Number of Panels: {sizing_result.panel_count}")
        # Display information to the user
    print(f"Range: {sizing_result.panel_count_min} - {sizing_result.panel_count_max} panels")
        # Display information to the user
    
    print(f"\n BATTERY SYSTEM")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print(f"Capacity: {sizing_result.battery_capacity_kwh:.2f} kWh")
        # Display information to the user
    print(f"Number of Batteries: {sizing_result.battery_count}")
        # Display information to the user
    print(f"Chemistry: {user_preferences['battery_chemistry']}")
        # Display information to the user
    
    print(f"\n INVERTER & CONTROLLER")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print(f"Inverter Power: {sizing_result.inverter_power_watts:.0f}W")
        # Display information to the user
    print(f"Charge Controller: {sizing_result.charge_controller_current:.1f}A")
        # Display information to the user
    
    print(f"\n COST ESTIMATE")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print(f"Estimated Cost: ₦{sizing_result.estimated_cost_min:,.0f} - ₦{sizing_result.estimated_cost_max:,.0f}")
        # Display information to the user
    print(f"Cost per kWh: ₦{sizing_result.cost_per_kwh:,.0f}/kWh/year")
        # Display information to the user
    
    print(f"\n EDUCATIONAL BREAKDOWN")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    print("Calculation Steps:")
        # Display information to the user
    for i, step in enumerate(sizing_result.calculation_steps[:3], 1):
        # Go through each item in the list
        print(f"  {i}. {step['title']}: {step['result']}")
        # Display information to the user
    
    print(f"\n RECOMMENDATIONS")
        # Display information to the user
    print("-" * 30)
        # Display information to the user
    for i, rec in enumerate(sizing_result.recommendations[:3], 1):
        # Go through each item in the list
        print(f"  {i}. {rec}")
        # Display information to the user
    
    print(f"\n COMPLETE WORKFLOW SUCCESSFUL!")
        # Display information to the user
    print("The SystemSizingAgent successfully calculated your solar system requirements!")
        # Display information to the user
    
    return True
        # Send the result back to the caller

def demo_user_scenarios():
    # This function performs a specific task for the solar system
    """Demo different user scenarios"""
    print("\n TESTING DIFFERENT USER SCENARIOS")
        # Display information to the user
    print("=" * 50)
        # Display information to the user
    
    scenarios = [
        {
            "name": "Small Apartment",
            "appliances": ["fridge", "lights", "fan", "phone charger"],
            "location": {"city": "Lagos", "state": "Lagos", "region": "South West"},
            "preferences": {"autonomy_days": 2, "battery_chemistry": "LiFePO4", "budget_range": "budget"}
        },
        {
            "name": "Medium House", 
            "appliances": ["fridge", "television", "lights", "fan", "washing machine", "microwave"],
            "location": {"city": "Abuja", "state": "FCT", "region": "North Central"},
            "preferences": {"autonomy_days": 3, "battery_chemistry": "LiFePO4", "budget_range": "medium"}
        },
        {
            "name": "Large Villa",
            "appliances": ["fridge", "television", "lights", "fan", "washing machine", "microwave", "air conditioner", "water pump"],
            "location": {"city": "Port Harcourt", "state": "Rivers", "region": "South South"},
            "preferences": {"autonomy_days": 5, "battery_chemistry": "LiFePO4", "budget_range": "premium"}
        }
    ]
    
    # Initialize agents
    try:
        # Try to execute the code safely
        geo_agent = GeoAgent()
        sizing_agent = SystemSizingAgent()
    except Exception as e:
        # Handle any errors that might occur
        print(f" Error initializing agents: {e}")
        # Display information to the user
        return False
        # Send the result back to the caller
    
    for scenario in scenarios:
        # Go through each item in the list
        print(f"\n Testing: {scenario['name']}")
        # Display information to the user
        print("-" * 40)
        # Display information to the user
        
        try:
        # Try to execute the code safely
            # Simulate energy calculation (normally from InputMappingAgent)
            energy_kwh = len(scenario['appliances']) * 2.5  # Simulate 2.5 kWh per appliance
            
            # Process location
            location_intelligence = geo_agent.process_location(scenario['location'])
            
            if location_intelligence:
        # Check a condition
                # Calculate system sizing
                load_data = {'daily_energy_kwh': energy_kwh}
                result = sizing_agent.calculate_system_sizing(
                    load_data, location_intelligence, scenario['preferences']
                )
                
                print(f" {scenario['name']} calculated successfully")
        # Display information to the user
                print(f"   Energy: {energy_kwh:.1f} kWh/day")
        # Display information to the user
                print(f"   Panels: {result.panel_count} ({result.panel_power_watts:.0f}W)")
        # Display information to the user
                print(f"   Batteries: {result.battery_count} ({result.battery_capacity_kwh:.1f} kWh)")
        # Display information to the user
                print(f"   Cost: ₦{result.estimated_cost_min:,.0f} - ₦{result.estimated_cost_max:,.0f}")
        # Display information to the user
            else:
                print(f" Location processing failed for {scenario['name']}")
        # Display information to the user
                
        except Exception as e:
        # Handle any errors that might occur
            print(f" Error calculating {scenario['name']}: {e}")
        # Display information to the user
    
    return True
        # Send the result back to the caller

if __name__ == "__main__":
        # Check a condition
    print(" SystemSizingAgent Workflow Demo")
        # Display information to the user
    print("=" * 50)
        # Display information to the user
    
    print("1. Complete Workflow Demo")
        # Display information to the user
    print("2. Multiple User Scenarios")
        # Display information to the user
    
    choice = input("\nChoose demo (1-2): ") or "1"
    
    if choice == "1":
        # Check a condition
        success = demo_complete_workflow()
        if success:
        # Check a condition
            print("\n Complete workflow demo successful!")
        # Display information to the user
        else:
            print("\n Demo failed. Check errors above.")
        # Display information to the user
    elif choice == "2":
        success = demo_user_scenarios()
        if success:
        # Check a condition
            print("\n Multiple scenarios demo successful!")
        # Display information to the user
        else:
            print("\n Demo failed. Check errors above.")
        # Display information to the user
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")
        # Display information to the user
