"""
Base Agent Class for Solar Recommender System
Provides common functionality for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all agents in the solar recommender system"""
    
    def __init__(self, name: str, llm_manager=None, tool_manager=None, nlp_processor=None):
        self.name = name
        # Backwards-compatible alias
        self.agent_name = name
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.nlp_processor = nlp_processor
        self.logger = self._setup_logger()
        self.status = 'initialized'
        self.last_activity = datetime.now()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    def get_llm(self, task_type: str = 'general') -> Optional[Any]:
        """Get appropriate LLM for the task"""
        if self.llm_manager:
            return self.llm_manager.get_llm(task_type)
        return None
    
    def search_web(self, query: str) -> str:
        """Search the web for information"""
        if self.tool_manager:
            return self.tool_manager.search_web(query)
        return ""
    
    def scrape_website(self, url: str) -> Dict[str, Any]:
        """Scrape website for product information"""
        if self.tool_manager:
            return self.tool_manager.scrape_website(url)
        return {}
    
    def process_nlp(self, text: str) -> Dict[str, Any]:
        """Process text with NLP"""
        if self.nlp_processor:
            return self.nlp_processor.process_input(text)
        return {}
    
    def log_activity(self, message: str, level: str = 'info'):
        """Log agent activity"""
        if level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        self.last_activity = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': self.name,
            'status': self.status,
            'last_activity': self.last_activity.isoformat(),
            'llm_available': self.llm_manager is not None,
            'tools_available': self.tool_manager is not None,
            'nlp_available': self.nlp_processor is not None
        }
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data has required fields"""
        for field in required_fields:
            if field not in input_data:
                self.log_activity(f"Missing required field: {field}", 'error')
                return False
        return True
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_message = f"Error in {self.name}: {str(error)}"
        if context:
            error_message += f" (Context: {context})"
        
        self.log_activity(error_message, 'error')
        
        return {
            'success': False,
            'error': str(error),
            'agent': self.name,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_response(self, data: Any, success: bool = True, message: str = "") -> Dict[str, Any]:
        """Create standardized response"""
        return {
            'success': success,
            'data': data,
            'message': message,
            'agent': self.name,
            'timestamp': datetime.now().isoformat()
        }