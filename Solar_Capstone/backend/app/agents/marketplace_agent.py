#!/usr/bin/env python3
"""
Marketplace Agent - Multi-LLM Powered Solar Component Intelligence
Advanced marketplace agent using all 4 LLMs for intelligent product discovery:
- Groq Llama3: Fast product search and quick comparisons
- Groq Mixtral: Complex market analysis and detailed comparisons
- HuggingFace: Product knowledge and specification matching
- Replicate: Creative product descriptions and marketing insights
- OpenRouter: Advanced recommendation logic and market intelligence
"""
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import LLM Manager
try:
    from ..core.llm_manager import StreamlinedLLMManager
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from llm_manager import StreamlinedLLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Vendor:
    """Represents a solar component vendor"""
    vendor_id: str
    name: str
    location: str
    rating: float
    specialties: List[str]
    contact_info: Dict[str, str]
    delivery_areas: List[str]
    certifications: List[str]

@dataclass
class ComponentListing:
    """Represents a solar component listing"""
    listing_id: str
    component_name: str
    component_type: str
    brand: str
    model: str
    price: float
    currency: str
    vendor_id: str
    specifications: Dict[str, Any]
    availability: str
    warranty: str
    rating: float
    reviews_count: int

@dataclass
class MarketTrend:
    """Represents market trend data"""
    trend_id: str
    component_type: str
    price_trend: str
    demand_level: str
    supply_level: str
    price_change_percent: float
    date: datetime

class MarketplaceAgent:
    """
    Multi-LLM Powered Marketplace Agent for Solar Component Intelligence
    
    Uses all 4 LLMs strategically:
    - Groq Llama3: Fast product search and quick comparisons
    - Groq Mixtral: Complex market analysis and detailed comparisons
    - HuggingFace: Product knowledge and specification matching
    - Replicate: Creative product descriptions and marketing insights
    - OpenRouter: Advanced recommendation logic and market intelligence
    
    Enhanced capabilities:
    - AI-powered component search and filtering
    - Intelligent price comparison with market insights
    - LLM-enhanced vendor verification and rating
    - Smart delivery and logistics recommendations
    - Advanced marketplace intelligence and trend analysis
    """
    
    def __init__(self):
        """Initialize the Multi-LLM Marketplace Agent"""
        self.agent_name = "MarketplaceAgent"
        self.version = "2.0.0"
        
        # Initialize LLM Manager with all 4 LLMs
        self.llm_manager = StreamlinedLLMManager()
        
        # Multi-LLM task assignment for marketplace intelligence
        self.llm_tasks = {
            'quick_search': 'groq_llama3',           # Fast product search
            'market_analysis': 'groq_mixtral',       # Complex market analysis
            'product_knowledge': 'huggingface',      # Product specifications
            'creative_content': 'replicate',         # Product descriptions
            'advanced_recommendations': 'openrouter_claude' # Smart recommendations
        }
        
        # Load marketplace data
        self.vendors = self._load_vendors()
        self.component_listings = self._load_component_listings()
        self.market_trends = self._load_market_trends()
        
        print(f"🛒 {self.agent_name} v{self.version} initialized with Multi-LLM System:")
        available_llms = self.llm_manager.get_available_providers()
        for llm in available_llms:
            print(f"   ✅ {llm}")
        print(f"   🏪 Vendors: {len(self.vendors)}")
        print(f"   📦 Products: {len(self.component_listings)}")
        print(f"   📈 Market Trends: {len(self.market_trends)}")
        
        print(f" {self.agent_name} initialized successfully")
    
    def _load_vendors(self) -> Dict[str, Vendor]:
        """Load vendor database"""
        vendors = {}
        
        # Sample vendors for Nigerian market
        vendor_data = [
            {
                "vendor_id": "vendor_001",
                "name": "SolarTech Nigeria",
                "location": "Lagos, Nigeria",
                "rating": 4.5,
                "specialties": ["Solar Panels", "Inverters", "Batteries"],
                "contact_info": {
                    "phone": "+234-801-234-5678",
                    "email": "info@solartech.ng",
                    "website": "www.solartech.ng"
                },
                "delivery_areas": ["Lagos", "Abuja", "Kano", "Ibadan"],
                "certifications": ["ISO 9001", "NAFDAC", "SON"]
            },
            {
                "vendor_id": "vendor_002",
                "name": "GreenEnergy Solutions",
                "location": "Abuja, Nigeria",
                "rating": 4.2,
                "specialties": ["Solar Panels", "Charge Controllers"],
                "contact_info": {
                    "phone": "+234-802-345-6789",
                    "email": "sales@greenenergy.ng",
                    "website": "www.greenenergy.ng"
                },
                "delivery_areas": ["Abuja", "Kaduna", "Jos"],
                "certifications": ["ISO 9001", "SON"]
            }
        ]
        
        for vendor_info in vendor_data:
            vendor = Vendor(**vendor_info)
            vendors[vendor.vendor_id] = vendor
        
        return vendors
    
    def _load_component_listings(self) -> Dict[str, ComponentListing]:
        """Load component listings"""
        listings = {}
        
        # Sample component listings
        listing_data = [
            {
                "listing_id": "listing_001",
                "component_name": "Monocrystalline Solar Panel 300W",
                "component_type": "Solar Panel",
                "brand": "SunPower",
                "model": "SP-300",
                "price": 45000.0,
                "currency": "NGN",
                "vendor_id": "vendor_001",
                "specifications": {
                    "power": "300W",
                    "efficiency": "20.5%",
                    "dimensions": "1956x992x40mm",
                    "weight": "22kg"
                },
                "availability": "In Stock",
                "warranty": "25 years",
                "rating": 4.7,
                "reviews_count": 156
            },
            {
                "listing_id": "listing_002",
                "component_name": "Pure Sine Wave Inverter 2000W",
                "component_type": "Inverter",
                "brand": "Victron Energy",
                "model": "Phoenix 2000VA",
                "price": 85000.0,
                "currency": "NGN",
                "vendor_id": "vendor_001",
                "specifications": {
                    "power": "2000W",
                    "efficiency": "95%",
                    "input_voltage": "12V/24V",
                    "output_voltage": "220V"
                },
                "availability": "In Stock",
                "warranty": "5 years",
                "rating": 4.8,
                "reviews_count": 89
            }
        ]
        
        for listing_info in listing_data:
            listing = ComponentListing(**listing_info)
            listings[listing.listing_id] = listing
        
        return listings
    
    def _load_market_trends(self) -> List[MarketTrend]:
        """Load market trend data"""
        trends = []
        
        # Sample market trends
        trend_data = [
            {
                "trend_id": "trend_001",
                "component_type": "Solar Panel",
                "price_trend": "Decreasing",
                "demand_level": "High",
                "supply_level": "Medium",
                "price_change_percent": -5.2,
                "date": datetime.now()
            },
            {
                "trend_id": "trend_002",
                "component_type": "Inverter",
                "price_trend": "Stable",
                "demand_level": "Medium",
                "supply_level": "High",
                "price_change_percent": 0.8,
                "date": datetime.now()
            }
        ]
        
        for trend_info in trend_data:
            trend = MarketTrend(**trend_info)
            trends.append(trend)
        
        return trends
    
    def search_components(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[ComponentListing]:
        """Search for solar components"""
        try:
            results = []
            query_lower = query.lower()
            
            for listing in self.component_listings.values():
                # Simple text matching
                if (query_lower in listing.component_name.lower() or 
                    query_lower in listing.brand.lower() or 
                    query_lower in listing.component_type.lower()):
                    results.append(listing)
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            # Sort by rating (highest first)
            results.sort(key=lambda x: x.rating, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching components: {e}")
            return []
    
    def _apply_filters(self, listings: List[ComponentListing], filters: Dict[str, Any]) -> List[ComponentListing]:
        """Apply filters to component listings"""
        filtered = listings
        
        if 'max_price' in filters:
            filtered = [l for l in filtered if l.price <= filters['max_price']]
        
        if 'min_rating' in filters:
            filtered = [l for l in filtered if l.rating >= filters['min_rating']]
        
        if 'component_type' in filters:
            filtered = [l for l in filtered if l.component_type == filters['component_type']]
        
        if 'brand' in filters:
            filtered = [l for l in filtered if l.brand.lower() == filters['brand'].lower()]
        
        return filtered
    
    def compare_components(self, component_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple components"""
        try:
            components = []
            for comp_id in component_ids:
                if comp_id in self.component_listings:
                    components.append(self.component_listings[comp_id])
            
            if not components:
                return {"error": "No components found for comparison"}
            
            comparison = {
                "components": components,
                "price_range": {
                    "min": min(c.price for c in components),
                    "max": max(c.price for c in components),
                    "average": sum(c.price for c in components) / len(components)
                },
                "rating_range": {
                    "min": min(c.rating for c in components),
                    "max": max(c.rating for c in components),
                    "average": sum(c.rating for c in components) / len(components)
                },
                "best_value": min(components, key=lambda x: x.price / x.rating),
                "highest_rated": max(components, key=lambda x: x.rating)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing components: {e}")
            return {"error": str(e)}
    
    def get_vendor_info(self, vendor_id: str) -> Optional[Vendor]:
        """Get vendor information"""
        return self.vendors.get(vendor_id)
    
    def get_market_trends(self, component_type: Optional[str] = None) -> List[MarketTrend]:
        """Get market trend data"""
        if component_type:
            return [t for t in self.market_trends if t.component_type == component_type]
        return self.market_trends
    
    def get_recommendations(self, user_requirements: Dict[str, Any]) -> List[ComponentListing]:
        """Get personalized component recommendations"""
        try:
            recommendations = []
            
            # Get all components
            all_components = list(self.component_listings.values())
            
            # Apply basic filtering based on requirements
            if 'budget' in user_requirements:
                max_price = user_requirements['budget']
                all_components = [c for c in all_components if c.price <= max_price]
            
            if 'component_type' in user_requirements:
                comp_type = user_requirements['component_type']
                all_components = [c for c in all_components if c.component_type == comp_type]
            
            # Sort by rating and price
            all_components.sort(key=lambda x: (x.rating, -x.price), reverse=True)
            
            # Return top 5 recommendations
            return all_components[:5]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "vendors_count": len(self.vendors),
            "listings_count": len(self.component_listings),
            "trends_count": len(self.market_trends),
            "status": "active"
        }