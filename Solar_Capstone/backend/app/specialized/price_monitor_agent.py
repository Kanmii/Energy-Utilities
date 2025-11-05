"""
Price Monitor Agent - Real-time Price Tracking
Monitors solar component prices across multiple marketplaces
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import requests
from datetime import datetime, timedelta
import time
import threading

# Import core infrastructure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from agent_base import BaseAgent

@dataclass
class PriceAlert:
    """Price alert configuration"""
    product_id: str
    product_name: str
    current_price: float
    target_price: float
    alert_type: str # 'drop', 'rise', 'change'
    user_email: str
    is_active: bool

@dataclass
class PriceHistory:
    """Price history data point"""
    product_id: str
    price: float
    currency: str
    timestamp: datetime
    source: str
    availability: str

class PriceMonitorAgent(BaseAgent):
    """Price monitoring agent for solar components"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("PriceMonitorAgent", llm_manager, tool_manager, nlp_processor)
        self.price_history = {}
        self.price_alerts = []
        self.marketplace_sources = []
        self._load_price_data()
        self._start_monitoring()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process price monitoring request"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['action']):
                return self.create_response(None, False, "Missing action parameter")
            
            action = input_data['action']
            
            if action == 'get_prices':
                return self._get_current_prices(input_data)
            elif action == 'set_alert':
                return self._set_price_alert(input_data)
            elif action == 'get_price_history':
                return self._get_price_history(input_data)
            elif action == 'analyze_trends':
                return self._analyze_price_trends(input_data)
            else:
                return self.create_response(None, False, f"Unknown action: {action}")
            
        except Exception as e:
            return self.handle_error(e, "Price monitoring")
        finally:
            self.status = 'idle'
    
    def _get_current_prices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current prices for products"""
        try:
            product_query = input_data.get('product_query', '')
            category = input_data.get('category', 'all')
            
            # Get prices from all sources
            all_prices = []
            
            for source in self.marketplace_sources:
                try:
                    prices = self._scrape_prices_from_source(source, product_query, category)
                    all_prices.extend(prices)
                except Exception as e:
                    self.log_activity(f"Error scraping from {source}: {e}", 'warning')
                    continue
            
            # Aggregate and rank prices
            aggregated_prices = self._aggregate_prices(all_prices)
            
            return self.create_response(aggregated_prices, True, "Current prices retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting current prices")
    
    def _set_price_alert(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set price alert for a product"""
        try:
            product_id = input_data.get('product_id')
            target_price = input_data.get('target_price')
            alert_type = input_data.get('alert_type', 'drop')
            user_email = input_data.get('user_email')
            
            if not all([product_id, target_price, user_email]):
                return self.create_response(None, False, "Missing required parameters for price alert")
            
            # Create price alert
            alert = PriceAlert(
                product_id=product_id,
                product_name=input_data.get('product_name', 'Unknown'),
                current_price=input_data.get('current_price', 0),
                target_price=target_price,
                alert_type=alert_type,
                user_email=user_email,
                is_active=True
            )
            
            self.price_alerts.append(alert)
            
            return self.create_response({
                'alert_id': len(self.price_alerts) - 1,
                'message': 'Price alert set successfully'
            }, True, "Price alert created")
            
        except Exception as e:
            return self.handle_error(e, "Setting price alert")
    
    def _get_price_history(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get price history for a product"""
        try:
            product_id = input_data.get('product_id')
            days = input_data.get('days', 30)
            
            if not product_id:
                return self.create_response(None, False, "Product ID required")
            
            # Get price history
            if product_id in self.price_history:
                history = self.price_history[product_id]
                
                # Filter by date range
                cutoff_date = datetime.now() - timedelta(days=days)
                filtered_history = [
                    h for h in history 
                    if h.timestamp >= cutoff_date
                ]
                
                # Convert to dict for JSON serialization
                history_data = []
                for h in filtered_history:
                    history_data.append({
                        'price': h.price,
                        'currency': h.currency,
                        'timestamp': h.timestamp.isoformat(),
                        'source': h.source,
                        'availability': h.availability
                    })
                
                return self.create_response(history_data, True, "Price history retrieved")
            else:
                return self.create_response([], True, "No price history found")
            
        except Exception as e:
            return self.handle_error(e, "Getting price history")
    
    def _analyze_price_trends(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trends for products"""
        try:
            product_id = input_data.get('product_id')
            category = input_data.get('category', 'all')
            
            if product_id:
                # Analyze specific product
                trends = self._analyze_product_trends(product_id)
            else:
                # Analyze category trends
                trends = self._analyze_category_trends(category)
            
            return self.create_response(trends, True, "Price trends analyzed")
            
        except Exception as e:
            return self.handle_error(e, "Analyzing price trends")
    
    def _scrape_prices_from_source(self, source: str, product_query: str, category: str) -> List[Dict[str, Any]]:
        """Scrape prices from a specific source"""
        try:
            # Mock price scraping - in practice would use actual scraping
            mock_prices = [
                {
                    'product_id': f"{source}_1",
                    'product_name': f"Solar Panel 300W - {source}",
                    'price': np.random.randint(50000, 100000),
                    'currency': 'NGN',
                    'availability': 'available',
                    'source': source,
                    'timestamp': datetime.now()
                },
                {
                    'product_id': f"{source}_2",
                    'product_name': f"Battery 100Ah - {source}",
                    'price': np.random.randint(80000, 150000),
                    'currency': 'NGN',
                    'availability': 'available',
                    'source': source,
                    'timestamp': datetime.now()
                }
            ]
            
            return mock_prices
            
        except Exception as e:
            self.log_activity(f"Error scraping from {source}: {e}", 'warning')
            return []
    
    def _aggregate_prices(self, all_prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate prices from multiple sources"""
        try:
            # Group by product name
            product_groups = {}
            
            for price in all_prices:
                product_name = price['product_name']
                if product_name not in product_groups:
                    product_groups[product_name] = []
                product_groups[product_name].append(price)
            
            # Aggregate each product
            aggregated = []
            
            for product_name, prices in product_groups.items():
                if not prices:
                    continue
                
                # Calculate statistics
                price_values = [p['price'] for p in prices]
                min_price = min(price_values)
                max_price = max(price_values)
                avg_price = sum(price_values) / len(price_values)
                
                # Find best deal
                best_deal = min(prices, key=lambda x: x['price'])
                
                aggregated.append({
                    'product_name': product_name,
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_price': avg_price,
                    'best_deal': best_deal,
                    'source_count': len(prices),
                    'price_range': f"₦{min_price:,} - ₦{max_price:,}"
                })
            
            # Sort by average price
            aggregated.sort(key=lambda x: x['avg_price'])
            
            return aggregated
            
        except Exception as e:
            self.log_activity(f"Error aggregating prices: {e}", 'warning')
            return []
    
    def _analyze_product_trends(self, product_id: str) -> Dict[str, Any]:
        """Analyze price trends for a specific product"""
        try:
            if product_id not in self.price_history:
                return {'error': 'No price history found'}
            
            history = self.price_history[product_id]
            if len(history) < 2:
                return {'error': 'Insufficient price history'}
            
            # Calculate trends
            prices = [h.price for h in history]
            timestamps = [h.timestamp for h in history]
            
            # Price change
            price_change = prices[-1] - prices[0]
            price_change_pct = (price_change / prices[0]) * 100
            
            # Trend direction
            if len(prices) >= 7:
                recent_prices = prices[-7:]
                trend_direction = 'up' if recent_prices[-1] > recent_prices[0] else 'down'
            else:
                trend_direction = 'stable'
            
            # Volatility
            price_std = np.std(prices)
            volatility = (price_std / np.mean(prices)) * 100
            
            return {
                'product_id': product_id,
                'current_price': prices[-1],
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'data_points': len(history),
                'analysis_period': f"{timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing product trends: {e}", 'warning')
            return {'error': str(e)}
    
    def _analyze_category_trends(self, category: str) -> Dict[str, Any]:
        """Analyze price trends for a category"""
        try:
            # Get all products in category
            category_products = []
            
            for product_id, history in self.price_history.items():
                if len(history) > 0:
                    latest_price = history[-1]
                    category_products.append({
                        'product_id': product_id,
                        'current_price': latest_price.price,
                        'timestamp': latest_price.timestamp
                    })
            
            if not category_products:
                return {'error': 'No products found in category'}
            
            # Calculate category statistics
            prices = [p['current_price'] for p in category_products]
            
            return {
                'category': category,
                'product_count': len(category_products),
                'avg_price': np.mean(prices),
                'min_price': min(prices),
                'max_price': max(prices),
                'price_std': np.std(prices),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing category trends: {e}", 'warning')
            return {'error': str(e)}
    
    def _start_monitoring(self):
        """Start background price monitoring"""
        try:
            # Start monitoring thread
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
            
            self.log_activity("Price monitoring started")
            
        except Exception as e:
            self.log_activity(f"Error starting price monitoring: {e}", 'warning')
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                self._update_prices()
                time.sleep(3600) # Check every hour
            except Exception as e:
                self.log_activity(f"Error in monitoring loop: {e}", 'warning')
                time.sleep(3600)
    
    def _update_prices(self):
        """Update prices from all sources"""
        try:
            self.log_activity("Updating prices from all sources...")
            
            for source in self.marketplace_sources:
                try:
                    prices = self._scrape_prices_from_source(source, '', 'all')
                    
                    for price in prices:
                        product_id = price['product_id']
                        
                        if product_id not in self.price_history:
                            self.price_history[product_id] = []
                        
                        # Add new price point
                        price_point = PriceHistory(
                            product_id=product_id,
                            price=price['price'],
                            currency=price['currency'],
                            timestamp=price['timestamp'],
                            source=price['source'],
                            availability=price['availability']
                        )
                        
                        self.price_history[product_id].append(price_point)
                        
                        # Keep only last 30 days of history
                        cutoff_date = datetime.now() - timedelta(days=30)
                        self.price_history[product_id] = [
                            h for h in self.price_history[product_id]
                            if h.timestamp >= cutoff_date
                        ]
                    
                except Exception as e:
                    self.log_activity(f"Error updating prices from {source}: {e}", 'warning')
                    continue
            
            # Check price alerts
            self._check_price_alerts()
            
            self.log_activity("Price update completed")
            
        except Exception as e:
            self.log_activity(f"Error updating prices: {e}", 'warning')
    
    def _check_price_alerts(self):
        """Check and trigger price alerts"""
        try:
            for alert in self.price_alerts:
                if not alert.is_active:
                    continue
                
                if alert.product_id in self.price_history:
                    current_price = self.price_history[alert.product_id][-1].price
                    
                    # Check alert conditions
                    should_trigger = False
                    
                    if alert.alert_type == 'drop' and current_price <= alert.target_price:
                        should_trigger = True
                    elif alert.alert_type == 'rise' and current_price >= alert.target_price:
                        should_trigger = True
                    elif alert.alert_type == 'change':
                        price_change = abs(current_price - alert.current_price)
                        if price_change >= alert.target_price:
                            should_trigger = True
                    
                    if should_trigger:
                        self._trigger_price_alert(alert, current_price)
            
        except Exception as e:
            self.log_activity(f"Error checking price alerts: {e}", 'warning')
    
    def _trigger_price_alert(self, alert: PriceAlert, current_price: float):
        """Trigger a price alert"""
        try:
            # In practice, would send email notification
            self.log_activity(f"Price alert triggered for {alert.product_name}: ₦{current_price:,}")
            
            # Deactivate alert
            alert.is_active = False
            
        except Exception as e:
            self.log_activity(f"Error triggering price alert: {e}", 'warning')
    
    def _load_price_data(self):
        """Load existing price data"""
        try:
            # Initialize marketplace sources
            self.marketplace_sources = [
                'Jumia',
                'Konga', 
                'Jiji',
                'Solar Direct',
                'Green Energy Co'
            ]
            
            # Load price history if exists
            if os.path.exists('data/price_history.json'):
                with open('data/price_history.json', 'r') as f:
                    data = json.load(f)
                    # Convert back to PriceHistory objects
                    for product_id, history_data in data.items():
                        self.price_history[product_id] = [
                            PriceHistory(**h) for h in history_data
                        ]
            
            self.log_activity("Price data loaded successfully")
            
        except Exception as e:
            self.log_activity(f"Error loading price data: {e}", 'warning')
    
    def save_price_data(self):
        """Save price data to file"""
        try:
            # Convert PriceHistory objects to dict for JSON serialization
            data = {}
            for product_id, history in self.price_history.items():
                data[product_id] = [
                    {
                        'product_id': h.product_id,
                        'price': h.price,
                        'currency': h.currency,
                        'timestamp': h.timestamp.isoformat(),
                        'source': h.source,
                        'availability': h.availability
                    }
                    for h in history
                ]
            
            os.makedirs('data', exist_ok=True)
            with open('data/price_history.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            self.log_activity("Price data saved successfully")
            
        except Exception as e:
            self.log_activity(f"Error saving price data: {e}", 'warning')