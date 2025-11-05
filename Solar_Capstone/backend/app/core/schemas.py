# Data Models for the Solar System Platform
# These define what information the system expects from users
# and what information it sends back to users
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# ===== REQUEST SCHEMAS =====

class ApplianceInput(BaseModel):
    name: str = Field(..., description="Appliance name")
    category: str = Field(..., description="Appliance category")
    usage_hours: float = Field(..., ge=0, le=24, description="Daily usage hours")
    quantity: int = Field(1, ge=1, description="Quantity of appliances")

class LocationInput(BaseModel):
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State name")
    region: str = Field(..., description="Geographic region")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")

class SystemPreferences(BaseModel):
    autonomy_days: float = Field(3, ge=1, le=7, description="Backup days")
    battery_chemistry: str = Field("LiFePO4", description="Battery chemistry")
    budget_range: str = Field("medium", description="Budget range")
    quality_threshold: float = Field(0.8, ge=0.5, le=1.0, description="Quality threshold")

class SolarSystemRequest(BaseModel):
    appliances: List[ApplianceInput] = Field(..., description="List of appliances")
    location: LocationInput = Field(..., description="Location information")
    preferences: SystemPreferences = Field(..., description="System preferences")

# ===== RESPONSE SCHEMAS =====

class LocationData(BaseModel):
    location: str
    coordinates: Tuple[float, float]
    sun_peak_hours: float
    daily_radiation_kwh: float
    seasonal_variation: float
    confidence: float
    data_source: str

class SolarIrradianceData(BaseModel):
    location: str
    sun_peak_hours: float
    daily_radiation_kwh: float
    seasonal_variation: float
    confidence: float
    data_source: str

class WeatherContext(BaseModel):
    location: str
    temperature_range: Tuple[float, float]
    precipitation_mm: float
    wind_speed_kmh: float
    humidity_percent: float
    confidence: float
    data_source: str

class LocationIntelligence(BaseModel):
    location_data: LocationData
    solar_data: SolarIrradianceData
    weather_data: WeatherContext
    installation_factors: Dict[str, Any]
    regional_recommendations: List[str]

class SystemSizingResult(BaseModel):
    daily_energy_kwh: float
    daily_energy_kwh_min: float
    daily_energy_kwh_max: float
    panel_power_watts: float
    panel_power_watts_min: float
    panel_power_watts_max: float
    panel_count: int
    panel_count_min: int
    panel_count_max: int
    battery_capacity_kwh: float
    battery_capacity_kwh_min: float
    battery_capacity_kwh_max: float
    battery_count: int
    battery_count_min: int
    battery_count_max: int
    inverter_power_watts: float
    inverter_power_watts_min: float
    inverter_power_watts_max: float
    charge_controller_current: float
    charge_controller_current_min: float
    charge_controller_current_max: float
    system_efficiency: float
    autonomy_days: float
    backup_hours: float
    battery_chemistry: str
    estimated_cost_min: float
    estimated_cost_max: float
    cost_per_kwh: float
    
    # Engineering Factors
    derating_factors: Dict[str, float]
    safety_factors: Dict[str, float]
    installation_factors: Dict[str, Any]
    
    # Educational Breakdown
    calculation_steps: List[Dict[str, Any]]

class ComponentRecommendation(BaseModel):
    rank: int
    brand: str
    model: str
    component_type: str
    specifications: Dict[str, Any]
    price_range: Tuple[float, float]
    quality_score: float
    performance_score: float
    composite_score: float
    confidence: float
    recommendation_reason: str
    pros: List[str]
    cons: List[str]
    warranty_years: int
    availability: str
    market_position: str

class SystemRecommendation(BaseModel):
    system_id: str
    total_cost_range: Tuple[float, float]
    total_quality_score: float
    total_performance_score: float
    system_efficiency: float
    payback_period_years: float
    warranty_coverage: float
    components: Dict[str, List[ComponentRecommendation]]
    system_advantages: List[str]
    system_considerations: List[str]
    installation_notes: List[str]
    maintenance_requirements: List[str]

class SolarSystemResponse(BaseModel):
    success: bool
    message: str
    system_id: str
    timestamp: datetime
    location_intelligence: LocationIntelligence
    sizing_result: SystemSizingResult
    recommendations: SystemRecommendation
    processing_time: float

# ===== ERROR SCHEMAS =====

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: datetime

# ===== LEGACY SCHEMAS (for backward compatibility) =====

class UserRequest(BaseModel):
    budget_ngn: Optional[float] = Field(None, ge=0)
    address: Optional[str] = None
    state: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    autonomy_days: Optional[float] = Field(None, ge=0, le=7)
    panel_placement: Optional[str] = Field(None, description="roof, ground, balcony")
    load_profile_id: Optional[str] = None

class GeoContext(BaseModel):
    avg_sun_hours: Optional[float] = None
    is_rainy_season: Optional[bool] = None
    source: Optional[str] = None

class SizingResult(BaseModel):
    inverter_va: float
    battery_kwh: float
    panel_kw: float
    controller_a: float
    notes: Optional[str] = None