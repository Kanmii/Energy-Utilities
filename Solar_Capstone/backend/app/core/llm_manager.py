"""
Streamlined LLM Manager for Solar Recommender System
Focused on core providers: Groq, Cohere, HuggingFace, Replicate
"""

import os
import json
import logging
import random
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlinedLLMManager:
    """
    Streamlined LLM Manager for Solar Recommender System
    Focuses on core providers with fallback mechanisms
    """
    
    def __init__(self):
        """Initialize the LLM manager with available providers"""
        self.providers = {}
        self.available_providers = []
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        try:
            # Groq
            if os.getenv('GROQ_API_KEY'):
                self.providers['groq'] = {
                    'name': 'Groq',
                    'api_key': os.getenv('GROQ_API_KEY'),
                    'base_url': 'https://api.groq.com/openai/v1',
                    'models': ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768'],
                    'status': 'available'
                }
                self.available_providers.append('groq')
                logger.info("Groq provider initialized")
            else:
                logger.warning("Groq not available (missing API key)")
                
            # Cohere
            if os.getenv('COHERE_API_KEY'):
                self.providers['cohere'] = {
                    'name': 'Cohere',
                    'api_key': os.getenv('COHERE_API_KEY'),
                    'base_url': 'https://api.cohere.ai/v1',
                    'models': ['command', 'command-light', 'command-nightly'],
                    'status': 'available'
                }
                self.available_providers.append('cohere')
                logger.info("Cohere provider initialized")
            else:
                logger.warning("Cohere not available (missing API key)")
                
            # HuggingFace
            if os.getenv('HUGGINGFACE_API_KEY'):
                self.providers['huggingface'] = {
                    'name': 'HuggingFace',
                    'api_key': os.getenv('HUGGINGFACE_API_KEY'),
                    'base_url': 'https://api-inference.huggingface.co/models',
                    'models': ['microsoft/DialoGPT-medium', 'facebook/blenderbot-400M-distill'],
                    'status': 'available'
                }
                self.available_providers.append('huggingface')
                logger.info("HuggingFace provider initialized")
            else:
                logger.warning("HuggingFace not available (missing API key)")
                
            # Replicate
            if os.getenv('REPLICATE_API_TOKEN'):
                self.providers['replicate'] = {
                    'name': 'Replicate',
                    'api_key': os.getenv('REPLICATE_API_TOKEN'),
                    'base_url': 'https://api.replicate.com/v1',
                    'models': ['meta/llama-2-7b-chat', 'meta/llama-2-13b-chat'],
                    'status': 'available'
                }
                self.available_providers.append('replicate')
                logger.info("Replicate provider initialized")
            else:
                logger.warning("Replicate not available (missing API token)")
                
        except Exception as e:
            logger.error(f"Error initializing providers: {e}")
            
    def get_available_providers(self):
        """Get list of available providers"""
        return self.available_providers.copy()
        
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        return self.providers.get(provider_name, {})
        
    def select_provider(self, preferred_provider: str = None) -> str:
        """Select the best available provider"""
        if not self.available_providers:
            raise Exception("No LLM providers available")
            
        if preferred_provider and preferred_provider in self.available_providers:
            return preferred_provider
            
        # Priority order: Groq > Cohere > HuggingFace > Replicate
        priority_order = ['groq', 'cohere', 'huggingface', 'replicate']
        
        for provider in priority_order:
            if provider in self.available_providers:
                return provider
                
        # Fallback to first available
        return self.available_providers[0]
        
    def generate_response(self, prompt: str, provider: str = None, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the specified or best available provider
        
        Args:
            prompt: The input prompt
            provider: Preferred provider (optional)
            model: Preferred model (optional)
            **kwargs: Additional parameters for the LLM call
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Select provider
            selected_provider = self.select_provider(provider)
            provider_info = self.get_provider_info(selected_provider)
            
            if not provider_info:
                raise Exception(f"Provider {selected_provider} not available")
                
            # Select model
            if not model:
                model = provider_info['models'][0]
                
            # Generate response based on provider
            if selected_provider == 'groq':
                return self._call_groq(prompt, model, **kwargs)
            elif selected_provider == 'cohere':
                return self._call_cohere(prompt, model, **kwargs)
            elif selected_provider == 'huggingface':
                return self._call_huggingface(prompt, model, **kwargs)
            elif selected_provider == 'replicate':
                return self._call_replicate(prompt, model, **kwargs)
            else:
                raise Exception(f"Unsupported provider: {selected_provider}")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'provider': selected_provider if 'selected_provider' in locals() else None
            }
            
    def _call_groq(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Call Groq API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.providers["groq"]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000)
            }
            
            response = requests.post(
                f"{self.providers['groq']['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result['choices'][0]['message']['content'],
                    'provider': 'groq',
                    'model': model,
                    'usage': result.get('usage', {})
                }
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
            
    def _call_cohere(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Call Cohere API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.providers["cohere"]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': model,
                'prompt': prompt,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000)
            }
            
            response = requests.post(
                f"{self.providers['cohere']['base_url']}/generate",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result['generations'][0]['text'],
                    'provider': 'cohere',
                    'model': model
                }
            else:
                raise Exception(f"Cohere API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Cohere API call failed: {e}")
            raise
            
    def _call_huggingface(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Call HuggingFace API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.providers['huggingface']['api_key']}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'inputs': prompt,
                'parameters': {
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_length': kwargs.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(
                f"{self.providers['huggingface']['base_url']}/{model}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result[0]['generated_text'] if result else "No response generated",
                    'provider': 'huggingface',
                    'model': model
                }
            else:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"HuggingFace API call failed: {e}")
            raise
            
    def _call_replicate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Call Replicate API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Token {self.providers['replicate']['api_key']}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'input': {
                    'prompt': prompt,
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_length': kwargs.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(
                f"{self.providers['replicate']['base_url']}/predictions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('output', 'No response generated'),
                    'provider': 'replicate',
                    'model': model
                }
            else:
                raise Exception(f"Replicate API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Replicate API call failed: {e}")
            raise
            
    def get_status(self) -> Dict[str, Any]:
        """Get the status of all providers"""
        return {
            'available_providers': self.available_providers,
            'total_providers': len(self.providers),
            'available_count': len(self.available_providers),
            'providers': {name: info['status'] for name, info in self.providers.items()}
        }
        
    def test_provider(self, provider_name: str) -> bool:
        """Test if a provider is working"""
        try:
            test_prompt = "Hello, this is a test message."
            result = self.generate_response(test_prompt, provider=provider_name)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Provider test failed for {provider_name}: {e}")
            return False
            
    def get_llm(self, llm_type: str = 'generation'):
        """Get LLM for a specific type of task"""
        # Simplified wrapper for backward compatibility
        # Returns self since we handle all tasks through generate_response
        return self

    def invoke(self, prompt: str, **kwargs):
        """Direct invocation wrapper for backward compatibility"""
        result = self.generate_response(prompt, **kwargs)
        if result.get('success'):
            return result['response']
        return f"Error: {result.get('error', 'Unknown error')}"

# Global instance
llm_manager = StreamlinedLLMManager()