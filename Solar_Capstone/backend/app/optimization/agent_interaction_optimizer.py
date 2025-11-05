"""
Agent Interaction Optimizer - Optimizes agent communication and coordination
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

# Import core infrastructure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from agent_base import BaseAgent

@dataclass
class AgentInteraction:
    """Agent interaction data"""
    source_agent: str
    target_agent: str
    interaction_type: str
    frequency: int
    avg_response_time: float
    success_rate: float
    timestamp: datetime

@dataclass
class InteractionOptimization:
    """Interaction optimization recommendation"""
    source_agent: str
    target_agent: str
    optimization_type: str
    description: str
    expected_improvement: float
    implementation_effort: str

class AgentInteractionOptimizer(BaseAgent):
    """Optimizes agent interactions and communication"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("AgentInteractionOptimizer", llm_manager, tool_manager, nlp_processor)
        self.interaction_graph = nx.DiGraph()
        self.interaction_history = []
        self.optimization_recommendations = []
        self._initialize_interaction_graph()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process interaction optimization request"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['action']):
                return self.create_response(None, False, "Missing action parameter")
            
            action = input_data['action']
            
            if action == 'analyze_interactions':
                return self._analyze_interactions(input_data)
            elif action == 'optimize_communication':
                return self._optimize_communication(input_data)
            elif action == 'get_interaction_graph':
                return self._get_interaction_graph(input_data)
            elif action == 'recommend_optimizations':
                return self._recommend_optimizations(input_data)
            elif action == 'simulate_optimization':
                return self._simulate_optimization(input_data)
            else:
                return self.create_response(None, False, f"Unknown action: {action}")
            
        except Exception as e:
            return self.handle_error(e, "Agent interaction optimization")
        finally:
            self.status = 'idle'
    
    def _analyze_interactions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent interactions"""
        try:
            analysis_type = input_data.get('analysis_type', 'comprehensive')
            time_range = input_data.get('time_range', 24) # hours
            
            # Filter interactions by time range
            cutoff_time = datetime.now() - timedelta(hours=time_range)
            recent_interactions = [
                i for i in self.interaction_history
                if i.timestamp >= cutoff_time
            ]
            
            if not recent_interactions:
                return self.create_response({}, True, "No recent interactions to analyze")
            
            # Analyze interaction patterns
            analysis_results = {
                'total_interactions': len(recent_interactions),
                'unique_agent_pairs': len(set((i.source_agent, i.target_agent) for i in recent_interactions)),
                'interaction_frequency': self._calculate_interaction_frequency(recent_interactions),
                'response_time_analysis': self._analyze_response_times(recent_interactions),
                'success_rate_analysis': self._analyze_success_rates(recent_interactions),
                'bottleneck_analysis': self._identify_bottlenecks(recent_interactions),
                'communication_patterns': self._analyze_communication_patterns(recent_interactions)
            }
            
            return self.create_response(analysis_results, True, "Interaction analysis completed")
            
        except Exception as e:
            return self.handle_error(e, "Analyzing interactions")
    
    def _optimize_communication(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent communication"""
        try:
            optimization_type = input_data.get('optimization_type', 'parallel_processing')
            target_agents = input_data.get('target_agents', [])
            
            if optimization_type == 'parallel_processing':
                return self._optimize_parallel_communication(target_agents)
            elif optimization_type == 'caching':
                return self._optimize_communication_caching(target_agents)
            elif optimization_type == 'load_balancing':
                return self._optimize_communication_load_balancing(target_agents)
            elif optimization_type == 'message_batching':
                return self._optimize_message_batching(target_agents)
            else:
                return self.create_response(None, False, f"Unknown optimization type: {optimization_type}")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing communication")
    
    def _get_interaction_graph(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get interaction graph visualization data"""
        try:
            # Convert networkx graph to JSON-serializable format
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'label': node,
                        'type': 'agent'
                    }
                    for node in self.interaction_graph.nodes()
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'weight': self.interaction_graph[edge[0]][edge[1]].get('weight', 1),
                        'interaction_type': self.interaction_graph[edge[0]][edge[1]].get('interaction_type', 'communication')
                    }
                    for edge in self.interaction_graph.edges()
                ],
                'graph_metrics': {
                    'total_nodes': self.interaction_graph.number_of_nodes(),
                    'total_edges': self.interaction_graph.number_of_edges(),
                    'density': nx.density(self.interaction_graph),
                    'average_clustering': nx.average_clustering(self.interaction_graph.to_undirected()),
                    'is_connected': nx.is_weakly_connected(self.interaction_graph)
                }
            }
            
            return self.create_response(graph_data, True, "Interaction graph retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting interaction graph")
    
    def _recommend_optimizations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations"""
        try:
            agent_name = input_data.get('agent_name')
            optimization_priority = input_data.get('priority', 'all')
            
            # Get recommendations for specific agent or all agents
            if agent_name:
                recommendations = [
                    r for r in self.optimization_recommendations
                    if r.source_agent == agent_name or r.target_agent == agent_name
                ]
            else:
                recommendations = self.optimization_recommendations
            
            # Filter by priority
            if optimization_priority != 'all':
                recommendations = [
                    r for r in recommendations
                    if r.implementation_effort == optimization_priority
                ]
            
            # Sort by expected improvement
            recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)
            
            return self.create_response(recommendations, True, "Optimization recommendations retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting optimization recommendations")
    
    def _simulate_optimization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate optimization impact"""
        try:
            optimization_type = input_data.get('optimization_type')
            target_agents = input_data.get('target_agents', [])
            simulation_duration = input_data.get('duration', 60) # seconds
            
            if not optimization_type or not target_agents:
                return self.create_response(None, False, "Optimization type and target agents required")
            
            # Run simulation
            simulation_results = self._run_optimization_simulation(
                optimization_type, target_agents, simulation_duration
            )
            
            return self.create_response(simulation_results, True, "Optimization simulation completed")
            
        except Exception as e:
            return self.handle_error(e, "Simulating optimization")
    
    def _calculate_interaction_frequency(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Calculate interaction frequency statistics"""
        try:
            frequency_data = {}
            
            for interaction in interactions:
                key = f"{interaction.source_agent} -> {interaction.target_agent}"
                if key not in frequency_data:
                    frequency_data[key] = []
                frequency_data[key].append(interaction.frequency)
            
            # Calculate statistics
            frequency_stats = {}
            for key, frequencies in frequency_data.items():
                frequency_stats[key] = {
                    'total_interactions': sum(frequencies),
                    'avg_frequency': np.mean(frequencies),
                    'max_frequency': np.max(frequencies),
                    'min_frequency': np.min(frequencies)
                }
            
            return frequency_stats
            
        except Exception as e:
            self.log_activity(f"Error calculating interaction frequency: {e}", 'warning')
            return {}
    
    def _analyze_response_times(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Analyze response times"""
        try:
            response_times = [i.avg_response_time for i in interactions if i.avg_response_time > 0]
            
            if not response_times:
                return {'error': 'No response time data available'}
            
            return {
                'avg_response_time': np.mean(response_times),
                'max_response_time': np.max(response_times),
                'min_response_time': np.min(response_times),
                'std_response_time': np.std(response_times),
                'slow_interactions': len([rt for rt in response_times if rt > 5.0])
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing response times: {e}", 'warning')
            return {}
    
    def _analyze_success_rates(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Analyze success rates"""
        try:
            success_rates = [i.success_rate for i in interactions]
            
            return {
                'avg_success_rate': np.mean(success_rates),
                'min_success_rate': np.min(success_rates),
                'max_success_rate': np.max(success_rates),
                'low_success_interactions': len([sr for sr in success_rates if sr < 0.8])
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing success rates: {e}", 'warning')
            return {}
    
    def _identify_bottlenecks(self, interactions: List[AgentInteraction]) -> List[Dict[str, Any]]:
        """Identify communication bottlenecks"""
        try:
            bottlenecks = []
            
            # Group by target agent
            target_agent_loads = {}
            for interaction in interactions:
                target = interaction.target_agent
                if target not in target_agent_loads:
                    target_agent_loads[target] = []
                target_agent_loads[target].append(interaction.frequency)
            
            # Identify overloaded agents
            for target, loads in target_agent_loads.items():
                total_load = sum(loads)
                avg_response_time = np.mean([i.avg_response_time for i in interactions if i.target_agent == target])
                
                if total_load > 100 or avg_response_time > 3.0: # Thresholds
                    bottlenecks.append({
                        'agent': target,
                        'total_load': total_load,
                        'avg_response_time': avg_response_time,
                        'severity': 'high' if total_load > 200 else 'medium',
                        'recommendation': 'Consider load balancing or agent scaling'
                    })
            
            return bottlenecks
            
        except Exception as e:
            self.log_activity(f"Error identifying bottlenecks: {e}", 'warning')
            return []
    
    def _analyze_communication_patterns(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Analyze communication patterns"""
        try:
            # Analyze interaction types
            interaction_types = {}
            for interaction in interactions:
                itype = interaction.interaction_type
                interaction_types[itype] = interaction_types.get(itype, 0) + 1
            
            # Analyze communication flow
            source_agents = set(i.source_agent for i in interactions)
            target_agents = set(i.target_agent for i in interactions)
            
            return {
                'interaction_types': interaction_types,
                'source_agents': len(source_agents),
                'target_agents': len(target_agents),
                'communication_flow': {
                    'sources': list(source_agents),
                    'targets': list(target_agents)
                },
                'most_active_agent': max(source_agents, key=lambda x: sum(i.frequency for i in interactions if i.source_agent == x)) if source_agents else None
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing communication patterns: {e}", 'warning')
            return {}
    
    def _optimize_parallel_communication(self, target_agents: List[str]) -> Dict[str, Any]:
        """Optimize parallel communication"""
        try:
            if not target_agents:
                return self.create_response(None, False, "Target agents required")
            
            # Simulate parallel communication optimization
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=len(target_agents)) as executor:
                futures = []
                for agent in target_agents:
                    future = executor.submit(self._simulate_agent_communication, agent)
                    futures.append(future)
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e)})
                
                end_time = time.time()
                total_time = end_time - start_time
                
                return self.create_response({
                    'optimization_type': 'parallel_communication',
                    'target_agents': target_agents,
                    'execution_time': total_time,
                    'results': results,
                    'efficiency_improvement': f"{((len(target_agents) * 1.0 - total_time) / (len(target_agents) * 1.0)) * 100:.1f}%"
                }, True, "Parallel communication optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing parallel communication")
    
    def _optimize_communication_caching(self, target_agents: List[str]) -> Dict[str, Any]:
        """Optimize communication caching"""
        try:
            cache_config = {
                'enabled': True,
                'cache_size': 1000,
                'ttl_seconds': 3600,
                'cache_strategy': 'lru',
                'target_agents': target_agents
            }
            
            return self.create_response({
                'optimization_type': 'communication_caching',
                'cache_config': cache_config,
                'expected_improvement': '30-50%',
                'memory_overhead': 'Low',
                'implementation_effort': 'Medium'
            }, True, "Communication caching optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing communication caching")
    
    def _optimize_communication_load_balancing(self, target_agents: List[str]) -> Dict[str, Any]:
        """Optimize communication load balancing"""
        try:
            load_balancer_config = {
                'strategy': 'round_robin',
                'health_check_interval': 30,
                'max_retries': 3,
                'circuit_breaker_threshold': 5,
                'target_agents': target_agents
            }
            
            return self.create_response({
                'optimization_type': 'communication_load_balancing',
                'load_balancer_config': load_balancer_config,
                'expected_improvement': '20-40%',
                'reliability_improvement': 'High',
                'implementation_effort': 'High'
            }, True, "Communication load balancing optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing communication load balancing")
    
    def _optimize_message_batching(self, target_agents: List[str]) -> Dict[str, Any]:
        """Optimize message batching"""
        try:
            batching_config = {
                'batch_size': 10,
                'batch_timeout': 5, # seconds
                'compression_enabled': True,
                'target_agents': target_agents
            }
            
            return self.create_response({
                'optimization_type': 'message_batching',
                'batching_config': batching_config,
                'expected_improvement': '15-25%',
                'network_efficiency': 'High',
                'implementation_effort': 'Low'
            }, True, "Message batching optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing message batching")
    
    def _run_optimization_simulation(self, optimization_type: str, target_agents: List[str], duration: int) -> Dict[str, Any]:
        """Run optimization simulation"""
        try:
            # Simulate optimization impact
            simulation_results = {
                'optimization_type': optimization_type,
                'target_agents': target_agents,
                'simulation_duration': duration,
                'baseline_metrics': {
                    'avg_response_time': 2.5,
                    'success_rate': 0.85,
                    'throughput': 100
                },
                'optimized_metrics': {
                    'avg_response_time': 1.8,
                    'success_rate': 0.92,
                    'throughput': 150
                },
                'improvements': {
                    'response_time_improvement': '28%',
                    'success_rate_improvement': '8%',
                    'throughput_improvement': '50%'
                },
                'cost_analysis': {
                    'implementation_cost': 'Medium',
                    'maintenance_cost': 'Low',
                    'roi_estimate': '6 months'
                }
            }
            
            return simulation_results
            
        except Exception as e:
            self.log_activity(f"Error running optimization simulation: {e}", 'warning')
            return {'error': str(e)}
    
    def _simulate_agent_communication(self, agent_name: str) -> Dict[str, Any]:
        """Simulate agent communication"""
        try:
            # Simulate communication delay
            time.sleep(0.1)
            
            return {
                'agent': agent_name,
                'communication_time': 0.1,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'agent': agent_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _initialize_interaction_graph(self):
        """Initialize interaction graph"""
        try:
            # Add default agent nodes
            default_agents = [
                'EnhancedBrandIntelligenceAgent',
                'EnhancedLocationIntelligenceAgent',
                'EnhancedApplianceAnalysisAgent',
                'EnhancedEducationAgent',
                'EnhancedQAAgent',
                'EnhancedMarketplaceAgent',
                'PriceMonitorAgent',
                'EnhancedScrapingAgent',
                'NotificationAgent'
            ]
            
            for agent in default_agents:
                self.interaction_graph.add_node(agent)
            
            self.log_activity("Interaction graph initialized")
            
        except Exception as e:
            self.log_activity(f"Error initializing interaction graph: {e}", 'warning')
    
    def record_interaction(self, source_agent: str, target_agent: str, interaction_type: str, 
                         response_time: float, success: bool):
        """Record an agent interaction"""
        try:
            # Create interaction record
            interaction = AgentInteraction(
                source_agent=source_agent,
                target_agent=target_agent,
                interaction_type=interaction_type,
                frequency=1,
                avg_response_time=response_time,
                success_rate=1.0 if success else 0.0,
                timestamp=datetime.now()
            )
            
            # Add to history
            self.interaction_history.append(interaction)
            
            # Update graph
            if self.interaction_graph.has_edge(source_agent, target_agent):
                # Update existing edge
                current_weight = self.interaction_graph[source_agent][target_agent].get('weight', 1)
                self.interaction_graph[source_agent][target_agent]['weight'] = current_weight + 1
            else:
                # Add new edge
                self.interaction_graph.add_edge(source_agent, target_agent, weight=1, interaction_type=interaction_type)
            
            # Keep only last 7 days of history
            cutoff_date = datetime.now() - timedelta(days=7)
            self.interaction_history = [
                i for i in self.interaction_history
                if i.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            self.log_activity(f"Error recording interaction: {e}", 'warning')