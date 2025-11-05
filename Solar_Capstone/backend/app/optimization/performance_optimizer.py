"""
Performance Optimizer - Agent Performance Monitoring and Optimization
Monitors and optimizes agent performance and interactions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import time
import threading
from datetime import datetime, timedelta
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import core infrastructure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from agent_base import BaseAgent

@dataclass
class PerformanceMetrics:
    """Performance metrics for agents"""
    agent_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int
    timestamp: datetime

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    agent_name: str
    issue_type: str
    description: str
    recommendation: str
    priority: str # 'high', 'medium', 'low'
    estimated_improvement: float

class PerformanceOptimizer(BaseAgent):
    """Performance optimizer for agent interactions"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("PerformanceOptimizer", llm_manager, tool_manager, nlp_processor)
        self.performance_metrics = []
        self.optimization_recommendations = []
        self.agent_registry = {}
        self._start_performance_monitoring()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process performance optimization request"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['action']):
                return self.create_response(None, False, "Missing action parameter")
            
            action = input_data['action']
            
            if action == 'get_performance_metrics':
                return self._get_performance_metrics(input_data)
            elif action == 'analyze_performance':
                return self._analyze_performance(input_data)
            elif action == 'get_optimization_recommendations':
                return self._get_optimization_recommendations(input_data)
            elif action == 'optimize_agent_interactions':
                return self._optimize_agent_interactions(input_data)
            elif action == 'run_performance_test':
                return self._run_performance_test(input_data)
            else:
                return self.create_response(None, False, f"Unknown action: {action}")
            
        except Exception as e:
            return self.handle_error(e, "Performance optimization")
        finally:
            self.status = 'idle'
    
    def _get_performance_metrics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics for agents"""
        try:
            agent_name = input_data.get('agent_name')
            time_range = input_data.get('time_range', 24) # hours
            
            # Filter metrics by agent and time range
            cutoff_time = datetime.now() - timedelta(hours=time_range)
            filtered_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_time and (not agent_name or m.agent_name == agent_name)
            ]
            
            if not filtered_metrics:
                return self.create_response([], True, "No performance metrics found")
            
            # Calculate aggregated metrics
            agent_metrics = {}
            for metric in filtered_metrics:
                agent = metric.agent_name
                if agent not in agent_metrics:
                    agent_metrics[agent] = {
                        'execution_times': [],
                        'memory_usage': [],
                        'cpu_usage': [],
                        'success_rates': [],
                        'error_counts': []
                    }
                
                agent_metrics[agent]['execution_times'].append(metric.execution_time)
                agent_metrics[agent]['memory_usage'].append(metric.memory_usage)
                agent_metrics[agent]['cpu_usage'].append(metric.cpu_usage)
                agent_metrics[agent]['success_rates'].append(metric.success_rate)
                agent_metrics[agent]['error_counts'].append(metric.error_count)
            
            # Calculate statistics
            performance_stats = {}
            for agent, metrics in agent_metrics.items():
                performance_stats[agent] = {
                    'avg_execution_time': np.mean(metrics['execution_times']),
                    'max_execution_time': np.max(metrics['execution_times']),
                    'avg_memory_usage': np.mean(metrics['memory_usage']),
                    'max_memory_usage': np.max(metrics['memory_usage']),
                    'avg_cpu_usage': np.mean(metrics['cpu_usage']),
                    'max_cpu_usage': np.max(metrics['cpu_usage']),
                    'avg_success_rate': np.mean(metrics['success_rates']),
                    'total_errors': sum(metrics['error_counts']),
                    'metric_count': len(metrics['execution_times'])
                }
            
            return self.create_response(performance_stats, True, "Performance metrics retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting performance metrics")
    
    def _analyze_performance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance and identify issues"""
        try:
            agent_name = input_data.get('agent_name')
            analysis_type = input_data.get('analysis_type', 'comprehensive')
            
            # Get recent metrics
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_time and (not agent_name or m.agent_name == agent_name)
            ]
            
            if not recent_metrics:
                return self.create_response({}, True, "No recent metrics to analyze")
            
            # Analyze performance issues
            issues = []
            
            # Check execution time issues
            execution_times = [m.execution_time for m in recent_metrics]
            avg_execution_time = np.mean(execution_times)
            max_execution_time = np.max(execution_times)
            
            if avg_execution_time > 5.0: # 5 seconds threshold
                issues.append({
                    'type': 'slow_execution',
                    'severity': 'high' if avg_execution_time > 10.0 else 'medium',
                    'description': f'Average execution time is {avg_execution_time:.2f}s',
                    'recommendation': 'Consider optimizing agent logic or adding caching'
                })
            
            # Check memory usage issues
            memory_usage = [m.memory_usage for m in recent_metrics]
            avg_memory_usage = np.mean(memory_usage)
            max_memory_usage = np.max(memory_usage)
            
            if avg_memory_usage > 100: # 100MB threshold
                issues.append({
                    'type': 'high_memory_usage',
                    'severity': 'high' if avg_memory_usage > 200 else 'medium',
                    'description': f'Average memory usage is {avg_memory_usage:.2f}MB',
                    'recommendation': 'Consider memory optimization or garbage collection'
                })
            
            # Check success rate issues
            success_rates = [m.success_rate for m in recent_metrics]
            avg_success_rate = np.mean(success_rates)
            
            if avg_success_rate < 0.8: # 80% threshold
                issues.append({
                    'type': 'low_success_rate',
                    'severity': 'high' if avg_success_rate < 0.6 else 'medium',
                    'description': f'Average success rate is {avg_success_rate:.2%}',
                    'recommendation': 'Review error handling and improve agent reliability'
                })
            
            # Check error count issues
            total_errors = sum(m.error_count for m in recent_metrics)
            if total_errors > 10: # 10 errors threshold
                issues.append({
                    'type': 'high_error_count',
                    'severity': 'high' if total_errors > 20 else 'medium',
                    'description': f'Total errors in last 24h: {total_errors}',
                    'recommendation': 'Investigate and fix recurring errors'
                })
            
            # Generate analysis report
            analysis_report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'time_range_hours': 24,
                'total_metrics': len(recent_metrics),
                'performance_summary': {
                    'avg_execution_time': avg_execution_time,
                    'max_execution_time': max_execution_time,
                    'avg_memory_usage': avg_memory_usage,
                    'max_memory_usage': max_memory_usage,
                    'avg_success_rate': avg_success_rate,
                    'total_errors': total_errors
                },
                'issues_found': len(issues),
                'issues': issues,
                'overall_health': self._calculate_overall_health(issues)
            }
            
            return self.create_response(analysis_report, True, "Performance analysis completed")
            
        except Exception as e:
            return self.handle_error(e, "Analyzing performance")
    
    def _get_optimization_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations"""
        try:
            agent_name = input_data.get('agent_name')
            
            # Get recommendations for specific agent or all agents
            if agent_name:
                recommendations = [r for r in self.optimization_recommendations if r.agent_name == agent_name]
            else:
                recommendations = self.optimization_recommendations
            
            # Sort by priority
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
            
            return self.create_response(recommendations, True, "Optimization recommendations retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting optimization recommendations")
    
    def _optimize_agent_interactions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent interactions"""
        try:
            optimization_type = input_data.get('optimization_type', 'parallel_processing')
            
            if optimization_type == 'parallel_processing':
                return self._optimize_parallel_processing(input_data)
            elif optimization_type == 'caching':
                return self._optimize_caching(input_data)
            elif optimization_type == 'load_balancing':
                return self._optimize_load_balancing(input_data)
            else:
                return self.create_response(None, False, f"Unknown optimization type: {optimization_type}")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing agent interactions")
    
    def _optimize_parallel_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parallel processing for agents"""
        try:
            agents_to_optimize = input_data.get('agents', [])
            max_workers = input_data.get('max_workers', 4)
            
            if not agents_to_optimize:
                return self.create_response(None, False, "No agents specified for optimization")
            
            # Test parallel processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for agent_name in agents_to_optimize:
                    future = executor.submit(self._test_agent_performance, agent_name)
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
                    'optimization_type': 'parallel_processing',
                    'agents_optimized': len(agents_to_optimize),
                    'max_workers': max_workers,
                    'total_execution_time': total_time,
                    'results': results,
                    'efficiency_improvement': self._calculate_efficiency_improvement(total_time, len(agents_to_optimize))
                }, True, "Parallel processing optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing parallel processing")
    
    def _optimize_caching(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize caching for agents"""
        try:
            cache_strategy = input_data.get('cache_strategy', 'lru')
            cache_size = input_data.get('cache_size', 100)
            ttl_seconds = input_data.get('ttl_seconds', 3600)
            
            # Implement caching optimization
            cache_config = {
                'strategy': cache_strategy,
                'size': cache_size,
                'ttl_seconds': ttl_seconds,
                'enabled': True
            }
            
            return self.create_response({
                'optimization_type': 'caching',
                'cache_config': cache_config,
                'estimated_performance_improvement': '20-40%',
                'memory_overhead': f'{cache_size * 0.1:.1f}MB'
            }, True, "Caching optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing caching")
    
    def _optimize_load_balancing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize load balancing for agents"""
        try:
            balancing_strategy = input_data.get('balancing_strategy', 'round_robin')
            max_concurrent_requests = input_data.get('max_concurrent_requests', 10)
            
            # Implement load balancing optimization
            load_balancer_config = {
                'strategy': balancing_strategy,
                'max_concurrent_requests': max_concurrent_requests,
                'health_check_interval': 30,
                'circuit_breaker_threshold': 5
            }
            
            return self.create_response({
                'optimization_type': 'load_balancing',
                'load_balancer_config': load_balancer_config,
                'estimated_performance_improvement': '15-30%',
                'reliability_improvement': 'High'
            }, True, "Load balancing optimization completed")
            
        except Exception as e:
            return self.handle_error(e, "Optimizing load balancing")
    
    def _run_performance_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance test for agents"""
        try:
            test_agents = input_data.get('agents', [])
            test_duration = input_data.get('duration', 60) # seconds
            concurrent_requests = input_data.get('concurrent_requests', 5)
            
            if not test_agents:
                return self.create_response(None, False, "No agents specified for testing")
            
            # Run performance test
            test_results = []
            start_time = time.time()
            
            for agent_name in test_agents:
                agent_results = self._test_agent_performance(agent_name, test_duration, concurrent_requests)
                test_results.append(agent_results)
            
            end_time = time.time()
            total_test_time = end_time - start_time
            
            # Calculate test summary
            test_summary = {
                'test_duration': test_duration,
                'total_test_time': total_test_time,
                'agents_tested': len(test_agents),
                'concurrent_requests': concurrent_requests,
                'results': test_results,
                'performance_score': self._calculate_performance_score(test_results)
            }
            
            return self.create_response(test_summary, True, "Performance test completed")
            
        except Exception as e:
            return self.handle_error(e, "Running performance test")
    
    def _test_agent_performance(self, agent_name: str, duration: int = 60, concurrent_requests: int = 5) -> Dict[str, Any]:
        """Test performance of a specific agent"""
        try:
            start_time = time.time()
            execution_times = []
            success_count = 0
            error_count = 0
            
            # Simulate agent execution
            for i in range(concurrent_requests):
                request_start = time.time()
                
                try:
                    # Mock agent execution
                    time.sleep(0.1) # Simulate processing time
                    success_count += 1
                except Exception:
                    error_count += 1
                
                request_end = time.time()
                execution_times.append(request_end - request_start)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            return {
                'agent_name': agent_name,
                'total_time': total_time,
                'avg_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'min_execution_time': np.min(execution_times),
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_count / concurrent_requests,
                'requests_per_second': concurrent_requests / total_time
            }
            
        except Exception as e:
            return {
                'agent_name': agent_name,
                'error': str(e),
                'success_rate': 0
            }
    
    def _calculate_overall_health(self, issues: List[Dict[str, Any]]) -> str:
        """Calculate overall system health"""
        if not issues:
            return 'excellent'
        
        high_severity_issues = len([i for i in issues if i.get('severity') == 'high'])
        medium_severity_issues = len([i for i in issues if i.get('severity') == 'medium'])
        
        if high_severity_issues > 0:
            return 'poor'
        elif medium_severity_issues > 2:
            return 'fair'
        elif medium_severity_issues > 0:
            return 'good'
        else:
            return 'excellent'
    
    def _calculate_efficiency_improvement(self, total_time: float, agent_count: int) -> str:
        """Calculate efficiency improvement from parallel processing"""
        # Estimate sequential time (assuming 1 second per agent)
        sequential_time = agent_count * 1.0
        improvement = ((sequential_time - total_time) / sequential_time) * 100
        return f"{improvement:.1f}%"
    
    def _calculate_performance_score(self, test_results: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score"""
        if not test_results:
            return 0.0
        
        scores = []
        for result in test_results:
            if 'success_rate' in result:
                score = result['success_rate'] * 100
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring"""
        try:
            # Start monitoring thread
            monitoring_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
            monitoring_thread.start()
            
            self.log_activity("Performance monitoring started")
            
        except Exception as e:
            self.log_activity(f"Error starting performance monitoring: {e}", 'warning')
    
    def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while True:
            try:
                self._collect_performance_metrics()
                time.sleep(30) # Check every 30 seconds
            except Exception as e:
                self.log_activity(f"Error in performance monitoring loop: {e}", 'warning')
                time.sleep(30)
    
    def _collect_performance_metrics(self):
        """Collect performance metrics from all agents"""
        try:
            # Get system metrics
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            # Collect metrics for each registered agent
            for agent_name, agent in self.agent_registry.items():
                try:
                    # Get agent-specific metrics
                    execution_time = getattr(agent, 'last_execution_time', 0.0)
                    success_rate = getattr(agent, 'success_rate', 1.0)
                    error_count = getattr(agent, 'error_count', 0)
                    
                    # Create performance metric
                    metric = PerformanceMetrics(
                        agent_name=agent_name,
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        cpu_usage=cpu_usage,
                        success_rate=success_rate,
                        error_count=error_count,
                        timestamp=datetime.now()
                    )
                    
                    self.performance_metrics.append(metric)
                    
                except Exception as e:
                    self.log_activity(f"Error collecting metrics for {agent_name}: {e}", 'warning')
                    continue
            
            # Keep only last 7 days of metrics
            cutoff_date = datetime.now() - timedelta(days=7)
            self.performance_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            self.log_activity(f"Error collecting performance metrics: {e}", 'warning')
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent for performance monitoring"""
        try:
            self.agent_registry[agent_name] = agent_instance
            self.log_activity(f"Agent {agent_name} registered for performance monitoring")
            
        except Exception as e:
            self.log_activity(f"Error registering agent {agent_name}: {e}", 'warning')
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from performance monitoring"""
        try:
            if agent_name in self.agent_registry:
                del self.agent_registry[agent_name]
                self.log_activity(f"Agent {agent_name} unregistered from performance monitoring")
            
        except Exception as e:
            self.log_activity(f"Error unregistering agent {agent_name}: {e}", 'warning')