#!/usr/bin/env python3
"""
Solar System MCP Server
Model Context Protocol server for solar system tools and data access
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    LoggingLevel
)

class SolarMCPServer:
    """MCP Server for Solar System Tools and Data Access"""
    
    def __init__(self):
        self.server = Server("solar-system-mcp")
        self.appliances_df = None
        self.components_df = {}
        self._setup_tools()
        self._setup_resources()
        self._load_data()
    
    def _setup_tools(self):
        """Setup MCP tools for solar system operations"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available solar system tools"""
            return [
                Tool(
                    name="calculate_solar_system",
                    description="Calculate solar system requirements based on appliances and location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "appliances": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "power_watts": {"type": "number"},
                                        "quantity": {"type": "integer"},
                                        "daily_hours": {"type": "number"}
                                    }
                                }
                            },
                            "location": {
                                "type": "object",
                                "properties": {
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"},
                                    "state": {"type": "string"}
                                }
                            },
                            "budget": {"type": "number"}
                        },
                        "required": ["appliances"]
                    }
                ),
                Tool(
                    name="get_appliance_data",
                    description="Get appliance information from database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "appliance_name": {"type": "string"},
                            "category": {"type": "string"}
                        }
                    }
                ),
                Tool(
                    name="get_component_recommendations",
                    description="Get component recommendations based on system requirements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "panel_power": {"type": "number"},
                            "battery_capacity": {"type": "number"},
                            "inverter_size": {"type": "number"},
                            "budget": {"type": "number"}
                        }
                    }
                ),
                Tool(
                    name="get_weather_data",
                    description="Get weather and solar irradiance data for location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                            "days": {"type": "integer", "default": 7}
                        },
                        "required": ["latitude", "longitude"]
                    }
                ),
                Tool(
                    name="estimate_costs",
                    description="Estimate solar system costs based on requirements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "panel_power": {"type": "number"},
                            "battery_capacity": {"type": "number"},
                            "inverter_size": {"type": "number"},
                            "location": {"type": "string"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "calculate_solar_system":
                    return await self._calculate_solar_system(arguments)
                elif name == "get_appliance_data":
                    return await self._get_appliance_data(arguments)
                elif name == "get_component_recommendations":
                    return await self._get_component_recommendations(arguments)
                elif name == "get_weather_data":
                    return await self._get_weather_data(arguments)
                elif name == "estimate_costs":
                    return await self._estimate_costs(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]
    
    def _setup_resources(self):
        """Setup MCP resources for data access"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available solar system resources"""
            return [
                Resource(
                    uri="solar://appliances",
                    name="Solar Appliances Database",
                    description="Database of appliances with power consumption data",
                    mimeType="application/json"
                ),
                Resource(
                    uri="solar://components",
                    name="Solar Components Database", 
                    description="Database of solar panels, batteries, inverters",
                    mimeType="application/json"
                ),
                Resource(
                    uri="solar://weather",
                    name="Weather Data",
                    description="Weather and solar irradiance data",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read solar system resources"""
            if uri == "solar://appliances":
                return self._get_appliances_json()
            elif uri == "solar://components":
                return self._get_components_json()
            elif uri == "solar://weather":
                return self._get_weather_json()
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    def _load_data(self):
        """Load solar system data from CSV files"""
        try:
            # Load appliances data
            self.appliances_df = pd.read_csv("data/interim/cleaned/appliances_cleaned.csv")
            print(f"Loaded {len(self.appliances_df)} appliances")
            
            # Load component data
            component_files = {
                'solar_panels': 'data/interim/cleaned/synthetic_solar_panels_synth.csv',
                'batteries': 'data/interim/cleaned/synthetic_batteries_synth.csv',
                'inverters': 'data/interim/cleaned/synthetic_inverters_synth.csv',
                'controllers': 'data/interim/cleaned/synthetic_charge_controllers_synth.csv'
            }
            
            for component_type, file_path in component_files.items():
                if os.path.exists(file_path):
                    self.components_df[component_type] = pd.read_csv(file_path)
                    print(f"Loaded {len(self.components_df[component_type])} {component_type}")
                    
        except Exception as e:
            print(f"Error loading data: {e}")
    
    async def _calculate_solar_system(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Calculate solar system requirements"""
        try:
            appliances = arguments.get('appliances', [])
            location = arguments.get('location', {})
            budget = arguments.get('budget')
            
            # Calculate total power and energy
            total_power = 0
            daily_energy = 0
            
            for app in appliances:
                power = app.get('power_watts', 0)
                quantity = app.get('quantity', 1)
                hours = app.get('daily_hours', 8)
                
                total_power += power * quantity
                daily_energy += (power * quantity * hours) / 1000
            
            # Calculate system requirements
            sun_hours = 5.5  # Nigeria average
            panel_power = daily_energy / sun_hours * 1.3  # 30% safety margin
            battery_capacity = daily_energy * 1.5  # 1.5 days autonomy
            inverter_size = total_power * 1.2  # 20% safety margin
            
            result = {
                "total_power_watts": total_power,
                "daily_energy_kwh": daily_energy,
                "recommended_panel_power": panel_power,
                "recommended_battery_capacity": battery_capacity,
                "recommended_inverter_size": inverter_size,
                "num_panels": int(panel_power / 400) + 1,
                "calculation_timestamp": datetime.now().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Calculation error: {str(e)}")]
    
    async def _get_appliance_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get appliance data from database"""
        try:
            appliance_name = arguments.get('appliance_name', '')
            category = arguments.get('category', '')
            
            if self.appliances_df is None:
                return [TextContent(type="text", text="Appliance database not loaded")]
            
            # Filter appliances
            filtered_df = self.appliances_df.copy()
            
            if appliance_name:
                filtered_df = filtered_df[
                    filtered_df['Appliance'].str.contains(appliance_name, case=False, na=False)
                ]
            
            if category:
                filtered_df = filtered_df[
                    filtered_df['Category'].str.contains(category, case=False, na=False)
                ]
            
            # Convert to JSON
            appliances_json = filtered_df.head(10).to_dict('records')
            
            return [TextContent(type="text", text=json.dumps(appliances_json, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting appliance data: {str(e)}")]
    
    async def _get_component_recommendations(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get component recommendations"""
        try:
            panel_power = arguments.get('panel_power', 0)
            battery_capacity = arguments.get('battery_capacity', 0)
            inverter_size = arguments.get('inverter_size', 0)
            budget = arguments.get('budget')
            
            recommendations = {}
            
            # Get solar panel recommendations
            if 'solar_panels' in self.components_df and panel_power > 0:
                panels_df = self.components_df['solar_panels']
                suitable_panels = panels_df[
                    (panels_df['rated_power_w'] >= panel_power * 0.8) & 
                    (panels_df['rated_power_w'] <= panel_power * 1.2)
                ].head(3)
                
                if not suitable_panels.empty:
                    recommendations['solar_panels'] = suitable_panels.to_dict('records')
            
            # Get battery recommendations
            if 'batteries' in self.components_df and battery_capacity > 0:
                batteries_df = self.components_df['batteries']
                suitable_batteries = batteries_df[
                    (batteries_df['capacity_kwh'] >= battery_capacity * 0.8) & 
                    (batteries_df['capacity_kwh'] <= battery_capacity * 1.5)
                ].head(3)
                
                if not suitable_batteries.empty:
                    recommendations['batteries'] = suitable_batteries.to_dict('records')
            
            # Get inverter recommendations
            if 'inverters' in self.components_df and inverter_size > 0:
                inverters_df = self.components_df['inverters']
                suitable_inverters = inverters_df[
                    (inverters_df['rated_power_w'] >= inverter_size * 0.8) & 
                    (inverters_df['rated_power_w'] <= inverter_size * 1.2)
                ].head(3)
                
                if not suitable_inverters.empty:
                    recommendations['inverters'] = suitable_inverters.to_dict('records')
            
            return [TextContent(type="text", text=json.dumps(recommendations, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting component recommendations: {str(e)}")]
    
    async def _get_weather_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get weather data for location"""
        try:
            latitude = arguments.get('latitude')
            longitude = arguments.get('longitude')
            days = arguments.get('days', 7)
            
            # This would integrate with your weather APIs
            # For now, return mock data
            weather_data = {
                "location": {"latitude": latitude, "longitude": longitude},
                "sun_peak_hours": 5.5,
                "average_temperature": 28.5,
                "humidity": 75,
                "irradiance": 4.8,
                "days_forecast": days,
                "timestamp": datetime.now().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(weather_data, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting weather data: {str(e)}")]
    
    async def _estimate_costs(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Estimate solar system costs"""
        try:
            panel_power = arguments.get('panel_power', 0)
            battery_capacity = arguments.get('battery_capacity', 0)
            inverter_size = arguments.get('inverter_size', 0)
            location = arguments.get('location', 'Nigeria')
            
            # Basic cost estimation
            panel_cost_per_watt = 150  # ₦150 per watt
            battery_cost_per_kwh = 200000  # ₦200,000 per kWh
            inverter_cost_per_watt = 200  # ₦200 per watt
            
            panel_cost = panel_power * panel_cost_per_watt
            battery_cost = battery_capacity * battery_cost_per_kwh
            inverter_cost = inverter_size * inverter_cost_per_watt
            installation_cost = (panel_cost + battery_cost + inverter_cost) * 0.2  # 20% installation
            
            total_cost = panel_cost + battery_cost + inverter_cost + installation_cost
            
            cost_breakdown = {
                "panel_cost": panel_cost,
                "battery_cost": battery_cost,
                "inverter_cost": inverter_cost,
                "installation_cost": installation_cost,
                "total_cost": total_cost,
                "currency": "NGN",
                "location": location,
                "timestamp": datetime.now().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(cost_breakdown, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error estimating costs: {str(e)}")]
    
    def _get_appliances_json(self) -> str:
        """Get appliances data as JSON"""
        if self.appliances_df is None:
            return json.dumps({"error": "Appliance database not loaded"})
        
        return self.appliances_df.head(50).to_json(orient='records', indent=2)
    
    def _get_components_json(self) -> str:
        """Get components data as JSON"""
        components_data = {}
        for component_type, df in self.components_df.items():
            components_data[component_type] = df.head(20).to_dict('records')
        
        return json.dumps(components_data, indent=2)
    
    def _get_weather_json(self) -> str:
        """Get weather data as JSON"""
        weather_data = {
            "nigeria_solar_data": {
                "average_sun_hours": 5.5,
                "peak_irradiance": 4.8,
                "seasonal_variation": 0.8,
                "best_months": ["March", "April", "May", "October", "November"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(weather_data, indent=2)
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="solar-system-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None
                    )
                )
            )

async def main():
    """Main entry point for the MCP server"""
    server = SolarMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
