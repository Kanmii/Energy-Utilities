"""
FastAPI Router - API Endpoints for Solar System Platform
This file defines all the web endpoints that users can access
It connects the web interface to all the AI agents
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
        # Import tools for creating web services
from fastapi.responses import JSONResponse
        # Import tools for creating web services
from typing import List, Dict, Any, Optional
        # Import tools needed for this file
import time
        # Import tools needed for this file
import uuid
        # Import tools needed for this file
from datetime import datetime
        # Import tools needed for this file
import traceback
        # Import tools needed for this file

# Import agents
from app.agents.input_mapping_agent import InputMappingAgent
        # Import tools needed for this file
from app.agents.location_intelligence_agent import LocationIntelligenceAgent
        # Import tools needed for this file
from app.agents.system_sizing_agent import EnhancedSystemSizingAgent
        # Import tools needed for this file
from app.agents.brand_intelligence_agent import BrandIntelligenceAgent
        # Import tools needed for this file
from app.agents.educational_agent import EducationalAgent
        # Import tools needed for this file

# Import schemas
from app.core.schemas import (
        # Import tools needed for this file
    SolarSystemRequest, SolarSystemResponse, ErrorResponse,
    LocationIntelligence, SystemSizingResult, SystemRecommendation,
    ApplianceInput, LocationInput, SystemPreferences
)

# Create router
api_router = APIRouter()

# ===== AGENT DEPENDENCIES =====

def get_input_mapping_agent():
    # This function gets information from the system
    """Dependency to get InputMappingAgent instance"""
    return InputMappingAgent()
        # Send the result back to the caller

def get_location_agent():
    # This function gets information from the system
    """Dependency to get LocationIntelligenceAgent instance"""
    return LocationIntelligenceAgent()
        # Send the result back to the caller

def get_sizing_agent():
    # This function gets information from the system
    """Dependency to get EnhancedSystemSizingAgent instance"""
    return EnhancedSystemSizingAgent()
        # Send the result back to the caller

def get_brand_agent():
    # This function gets information from the system
    """Dependency to get BrandIntelligenceAgent instance"""
    return BrandIntelligenceAgent()
        # Send the result back to the caller

def get_educational_agent():
    # This function gets information from the system
    """Dependency to get EducationalAgent instance"""
    return EducationalAgent()
        # Send the result back to the caller

# ===== HEALTH & STATUS ENDPOINTS =====

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        # Send the result back to the caller
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Solar System Recommendation Platform",
        "version": "1.0.0"
    }

@api_router.get("/status")
async def system_status():
    """System status with agent availability"""
    try:
        # Try to execute the code safely
        # Test agent initialization
        agents_status = {}
        
        try:
        # Try to execute the code safely
            input_agent = InputMappingAgent()
            agents_status["input_mapping"] = "ready"
        except Exception as e:
        # Handle any errors that might occur
            agents_status["input_mapping"] = f"error: {str(e)}"
        
        try:
        # Try to execute the code safely
            geo_agent = GeoAgent()
            agents_status["geo_agent"] = "ready"
        except Exception as e:
        # Handle any errors that might occur
            agents_status["geo_agent"] = f"error: {str(e)}"
        
        try:
        # Try to execute the code safely
            sizing_agent = EnhancedSystemSizingAgent()
            agents_status["system_sizing"] = "ready"
        except Exception as e:
        # Handle any errors that might occur
            agents_status["system_sizing"] = f"error: {str(e)}"
        
        try:
        # Try to execute the code safely
            brand_agent = BrandIntelligenceAgent()
            agents_status["brand_intelligence"] = "ready"
        except Exception as e:
        # Handle any errors that might occur
            agents_status["brand_intelligence"] = f"error: {str(e)}"
        
        try:
        # Try to execute the code safely
            educational_agent = EducationalAgent()
            agents_status["educational"] = "ready"
        except Exception as e:
        # Handle any errors that might occur
            agents_status["educational"] = f"error: {str(e)}"
        
        return {
        # Send the result back to the caller
            "status": "operational",
            "timestamp": datetime.now(),
            "agents": agents_status,
            "total_agents": len(agents_status),
            "ready_agents": len([s for s in agents_status.values() if s == "ready"])
        }
    except Exception as e:
        # Handle any errors that might occur
        return {
        # Send the result back to the caller
            "status": "error",
            "timestamp": datetime.now(),
            "error": str(e)
        }

# ===== INDIVIDUAL AGENT ENDPOINTS =====

@api_router.post("/agents/input-mapping/process")
async def process_appliances(
    appliances: List[ApplianceInput],
    input_agent: InputMappingAgent = Depends(get_input_mapping_agent)
):
    """Process appliances and calculate energy requirements"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Convert to agent format
        appliance_data = {
            'appliances': [app.name for app in appliances],
            'usage_hours': {app.name: app.usage_hours for app in appliances},
            'quantities': {app.name: app.quantity for app in appliances}
        }
        
        # Process with InputMappingAgent
        result = input_agent.process_appliances(appliance_data)
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Appliances processed successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error processing appliances: {str(e)}"
        )

@api_router.post("/agents/geo/process")
async def process_location(
    location: LocationInput,
    location_agent: LocationIntelligenceAgent = Depends(get_location_agent)
):
    """Process location and get solar intelligence"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Convert to agent format
        location_data = {
            'city': location.city,
            'state': location.state,
            'region': location.region,
            'latitude': location.latitude,
            'longitude': location.longitude
        }
        
        # Process with GeoAgent
        result = geo_agent.process_location(location_data)
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Location processed successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error processing location: {str(e)}"
        )

@api_router.post("/agents/system-sizing/calculate")
async def calculate_system_sizing(
    load_data: Dict[str, Any],
    location_intelligence: LocationIntelligence,
    preferences: SystemPreferences,
    sizing_agent: EnhancedSystemSizingAgent = Depends(get_sizing_agent)
):
    """Calculate system sizing requirements"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with SystemSizingAgent
        result = sizing_agent.calculate_system_sizing(
            load_data, location_intelligence, preferences.dict()
        )
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "System sizing calculated successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating system sizing: {str(e)}"
        )

@api_router.post("/agents/brand-intelligence/recommend")
async def recommend_components(
    system_requirements: Dict[str, Any],
    preferences: SystemPreferences,
    brand_agent: BrandIntelligenceAgent = Depends(get_brand_agent)
):
    """Get ML-powered component recommendations"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with BrandIntelligenceAgent
        result = brand_agent.recommend_components(
            system_requirements, preferences.dict()
        )
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Component recommendations generated successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@api_router.post("/agents/educational/guidance")
async def get_user_guidance(
    user_profile: Dict[str, Any],
    educational_agent: EducationalAgent = Depends(get_educational_agent)
):
    """Get personalized user guidance"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with EducationalAgent
        result = educational_agent.provide_user_guidance(user_profile)
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "User guidance generated successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error generating user guidance: {str(e)}"
        )

@api_router.post("/agents/educational/explain")
async def explain_calculation(
    process_type: str,
    data: Dict[str, Any],
    educational_agent: EducationalAgent = Depends(get_educational_agent)
):
    """Get step-by-step calculation explanations"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with EducationalAgent
        result = educational_agent.explain_calculation_process(process_type, data)
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Calculation explanation generated successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error generating calculation explanation: {str(e)}"
        )

@api_router.post("/agents/educational/content")
async def get_educational_content(
    topics: List[str],
    difficulty: str = "beginner",
    educational_agent: EducationalAgent = Depends(get_educational_agent)
):
    """Get educational content for specific topics"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with EducationalAgent
        result = educational_agent.get_educational_content(topics, difficulty)
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Educational content retrieved successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving educational content: {str(e)}"
        )

@api_router.post("/agents/educational/comprehensive")
async def get_comprehensive_educational_response(
    user_profile: Dict[str, Any],
    system_data: Dict[str, Any],
    context: str = "general",
    educational_agent: EducationalAgent = Depends(get_educational_agent)
):
    """Get comprehensive educational response"""
    try:
        # Try to execute the code safely
        start_time = time.time()
        
        # Process with EducationalAgent
        result = educational_agent.generate_educational_response(
            user_profile, system_data, context
        )
        
        processing_time = time.time() - start_time
        
        return {
        # Send the result back to the caller
            "success": True,
            "message": "Comprehensive educational response generated successfully",
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        # Handle any errors that might occur
        raise HTTPException(
            status_code=500,
            detail=f"Error generating comprehensive educational response: {str(e)}"
        )

# ===== INTEGRATED ENDPOINTS =====

@api_router.post("/solar-system/calculate", response_model=SolarSystemResponse)
async def calculate_solar_system(
    request: SolarSystemRequest,
    background_tasks: BackgroundTasks
):
    """
    Complete solar system calculation using all agents
    This is the main endpoint that orchestrates all agents
    """
    start_time = time.time()
    system_id = str(uuid.uuid4())
    
    try:
        # Try to execute the code safely
        # Initialize agents
        input_agent = InputMappingAgent()
        location_agent = LocationIntelligenceAgent()
        sizing_agent = EnhancedSystemSizingAgent()
        brand_agent = BrandIntelligenceAgent()
        
        # Step 1: Process appliances with InputMappingAgent
        appliance_data = {
            'appliances': [app.name for app in request.appliances],
            'usage_hours': {app.name: app.usage_hours for app in request.appliances},
            'quantities': {app.name: app.quantity for app in request.appliances}
        }
        
        load_result = input_agent.process_appliances(appliance_data)
        daily_energy = load_result.get('total_daily_energy_kwh', 0)
        
        # Step 2: Process location with GeoAgent
        location_data = {
            'city': request.location.city,
            'state': request.location.state,
            'region': request.location.region,
            'latitude': request.location.latitude,
            'longitude': request.location.longitude
        }
        
        location_intelligence = geo_agent.process_location(location_data)
        
        if not location_intelligence:
        # Check a condition
            raise HTTPException(
                status_code=400,
                detail="Failed to process location data"
            )
        
        # Step 3: Calculate system sizing
        load_data = {'daily_energy_kwh': daily_energy}
        sizing_result = sizing_agent.calculate_system_sizing(
            load_data, location_intelligence, request.preferences.dict()
        )
        
        # Step 4: Get component recommendations
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
            'budget_range': request.preferences.budget_range
        }
        
        recommendations = brand_agent.recommend_components(
            system_requirements, request.preferences.dict()
        )
        
        processing_time = time.time() - start_time
        
        # Create response
        response = SolarSystemResponse(
            success=True,
            message="Solar system calculation completed successfully",
            system_id=system_id,
            timestamp=datetime.now(),
            location_intelligence=location_intelligence,
            sizing_result=sizing_result,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
        # Log successful calculation
        background_tasks.add_task(
            log_calculation,
            system_id,
            request.dict(),
            response.dict()
        )
        
        return response
        # Send the result back to the caller
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle any errors that might occur
        processing_time = time.time() - start_time
        
        # Log error
        background_tasks.add_task(
            log_error,
            system_id,
            str(e),
            traceback.format_exc()
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating solar system: {str(e)}"
        )

@api_router.get("/solar-system/{system_id}")
async def get_system_result(system_id: str):
    """Get system calculation result by ID"""
    # This would typically query a database
    # For now, return a placeholder
    return {
        # Send the result back to the caller
        "message": "System result retrieval not implemented yet",
        "system_id": system_id,
        "note": "Results are currently returned directly from calculation endpoint"
    }

# ===== UTILITY ENDPOINTS =====

@api_router.get("/appliances/categories")
async def get_appliance_categories():
    """Get available appliance categories"""
    return {
        # Send the result back to the caller
        "categories": [
            "Kitchen", "Laundry", "Entertainment", "Lighting", "Cooling",
            "Heating", "Electronics", "Security", "Water Systems", "Other"
        ]
    }

@api_router.get("/locations/regions")
async def get_nigerian_regions():
    """Get Nigerian geographic regions"""
    return {
        # Send the result back to the caller
        "regions": [
            "South West", "South East", "South South",
            "North West", "North East", "North Central"
        ]
    }

@api_router.get("/components/types")
async def get_component_types():
    """Get available solar component types"""
    return {
        # Send the result back to the caller
        "component_types": [
            "solar_panels", "batteries", "inverters", "charge_controllers",
            "mounting_systems", "cables", "monitoring_systems"
        ]
    }

# ===== BACKGROUND TASKS =====

async def log_calculation(system_id: str, request_data: Dict, response_data: Dict):
    """Log successful calculation"""
    # This would typically save to a database
    print(f" System {system_id} calculated successfully")
        # Display information to the user

async def log_error(system_id: str, error: str, traceback: str):
    """Log calculation error"""
    # This would typically save to a database
    print(f" System {system_id} calculation failed: {error}")
        # Display information to the user

# ===== ERROR HANDLERS =====
# Note: Exception handlers are moved to main.py as APIRouter doesn't support them
