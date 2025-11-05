# Solar System MCP Integration

This directory contains the Model Context Protocol (MCP) integration for the Solar System project.

## What is MCP?

Model Context Protocol (MCP) is a standard for connecting AI models to external tools and data sources. It allows LLMs to access real-time data and perform complex operations through a standardized interface.

## Files

- `solar_mcp_server.py` - MCP server that provides solar system tools
- `solar_mcp_client.py` - MCP client for integrating with existing agents
- `start_mcp_server.py` - Script to start the MCP server
- `mcp_config.json` - Configuration for MCP tools and resources

## MCP Tools Available

### 1. calculate_solar_system
Calculate solar system requirements based on appliances and location.

**Input:**
- `appliances`: List of appliances with power consumption
- `location`: Geographic location data
- `budget`: Optional budget constraint

**Output:**
- Total power requirements
- Daily energy consumption
- Recommended system components
- Number of panels needed

### 2. get_appliance_data
Get appliance information from the database.

**Input:**
- `appliance_name`: Name of appliance to search for
- `category`: Appliance category filter

**Output:**
- List of matching appliances with specifications
- Power consumption data
- Usage recommendations

### 3. get_component_recommendations
Get component recommendations based on system requirements.

**Input:**
- `panel_power`: Required panel power
- `battery_capacity`: Required battery capacity
- `inverter_size`: Required inverter size
- `budget`: Optional budget constraint

**Output:**
- Recommended solar panels
- Recommended batteries
- Recommended inverters
- Price ranges and specifications

### 4. get_weather_data
Get weather and solar irradiance data for a location.

**Input:**
- `latitude`: Location latitude
- `longitude`: Location longitude
- `days`: Number of days to forecast

**Output:**
- Solar irradiance data
- Weather conditions
- Peak sun hours
- Seasonal variations

### 5. estimate_costs
Estimate solar system costs based on requirements.

**Input:**
- `panel_power`: Panel power requirement
- `battery_capacity`: Battery capacity requirement
- `inverter_size`: Inverter size requirement
- `location`: Installation location

**Output:**
- Component costs breakdown
- Installation costs
- Total system cost
- Cost per watt analysis

## MCP Resources Available

### 1. solar://appliances
Access to the appliances database with power consumption data.

### 2. solar://components
Access to the components database with solar panels, batteries, and inverters.

### 3. solar://weather
Access to weather and solar irradiance data.

## Integration with Existing Agents

The MCP client is integrated with the SuperAgent to provide enhanced capabilities:

- **Enhanced Calculations**: More accurate system sizing using MCP tools
- **Real-time Data**: Access to live weather and market data
- **Component Matching**: Better component recommendations based on real inventory
- **Cost Optimization**: More accurate cost estimates using current market data

## Usage

### Starting the MCP Server

```bash
cd backend/app/mcp
python start_mcp_server.py
```

### Using MCP in Agents

```python
from ..mcp.solar_mcp_client import get_mcp_client

# Get MCP client
mcp_client = await get_mcp_client()

# Use MCP tools
result = await mcp_client.calculate_solar_system(appliances, location, budget)
```

## Benefits of MCP Integration

1. **Standardized Interface**: Consistent way to access external tools and data
2. **Real-time Data**: Access to live weather, market, and component data
3. **Enhanced Accuracy**: More precise calculations using external data sources
4. **Scalability**: Easy to add new tools and data sources
5. **Interoperability**: Works with any MCP-compatible client

## Configuration

Edit `mcp_config.json` to:
- Enable/disable specific tools
- Configure data sources
- Set up resource access
- Customize tool behavior

## Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Check if all dependencies are installed
   - Verify CSV data files exist
   - Check server logs for errors

2. **Tools Not Available**
   - Ensure MCP server is running
   - Check tool configuration in mcp_config.json
   - Verify client connection

3. **Data Access Issues**
   - Verify CSV file paths in configuration
   - Check file permissions
   - Ensure data files are properly formatted

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export MCP_DEBUG=true
```

## Future Enhancements

- Integration with external APIs (weather, market data)
- Real-time component availability checking
- Advanced cost optimization algorithms
- Integration with installation scheduling systems
- Mobile app integration
