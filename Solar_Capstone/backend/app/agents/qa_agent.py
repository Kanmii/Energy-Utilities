#!/usr/bin/env python3
"""
Q&A Agent - Intelligent Question Answering
Advanced question answering with context awareness and knowledge retrieval
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
from datetime import datetime, timedelta
import re

# Import core infrastructure
import sys
import os
core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)
try:
    from agent_base import BaseAgent  # type: ignore
except ImportError:
    # Fallback for when agent_base is not available
    class BaseAgent:  # type: ignore
        def __init__(self, name, llm_manager=None, tool_manager=None, nlp_processor=None):
            self.name = name
            self.llm_manager = llm_manager
            self.tool_manager = tool_manager
            self.nlp_processor = nlp_processor
            self.status = 'initialized'
        
        def log_activity(self, message, level='info'):
            print(f"[{self.name}] {message}")
        
        def validate_input(self, input_data, required_fields):
            return all(field in input_data for field in required_fields)
        
        def create_response(self, data, success=True, message=""):
            return {'success': success, 'data': data, 'message': message}
        
        def handle_error(self, error, context=""):
            return {'success': False, 'error': str(error)}

@dataclass
class QAAnswer:
    """Enhanced Q&A answer with confidence and sources"""
    question: str
    answer: str
    confidence: float
    answer_type: str  # "direct", "inferred", "fallback"
    sources: List[str]
    related_questions: List[str]
    follow_up_suggestions: List[str]
    ai_explanation: str
    context_used: Dict[str, Any]

@dataclass
class KnowledgeEntry:
    """Knowledge base entry"""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    confidence: float
    source: str
    last_updated: datetime

class QAAgent(BaseAgent):
    """Advanced Q&A Agent with context awareness and knowledge retrieval"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("QAAgent", llm_manager, tool_manager, nlp_processor)
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []
        self.context_memory = {}
        self.answer_templates = self._load_answer_templates()
        
        print(f" {self.agent_name} initialized successfully")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Q&A request with enhanced capabilities"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['question']):
                return self.create_response(None, False, "Missing question parameter")
            
            question = input_data['question']
            context = input_data.get('context', {})
            user_profile = input_data.get('user_profile', {})
            
            # Analyze question
            question_analysis = self._analyze_question(question)
            
            # Retrieve relevant knowledge
            relevant_knowledge = self._retrieve_knowledge(question, question_analysis)
            
            # Generate answer
            answer = self._generate_answer(question, relevant_knowledge, context, user_profile)
            
            # Update conversation history
            self._update_conversation_history(question, answer)
            
            return self.create_response(answer, True, "Q&A processing completed successfully")
            
        except Exception as e:
            return self.handle_error(e, "Q&A processing")
        finally:
            self.status = 'idle'
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to understand intent and requirements"""
        try:
            # Basic question analysis
            question_lower = question.lower()
            
            # Determine question type
            question_type = self._classify_question_type(question_lower)
            
            # Extract key terms
            key_terms = self._extract_key_terms(question)
            
            # Determine complexity
            complexity = self._assess_complexity(question)
            
            # Check for follow-up indicators
            is_follow_up = self._is_follow_up_question(question)
            
            return {
                'question_type': question_type,
                'key_terms': key_terms,
                'complexity': complexity,
                'is_follow_up': is_follow_up,
                'original_question': question,
                'processed_question': self._preprocess_question(question)
            }
            
        except Exception as e:
            self.log_activity(f"Error analyzing question: {e}", 'error')
            return {
                'question_type': 'general',
                'key_terms': [],
                'complexity': 'medium',
                'is_follow_up': False,
                'original_question': question,
                'processed_question': question
            }
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        # Technical questions
        if any(word in question for word in ['how', 'what', 'why', 'calculate', 'formula']):
            return 'technical'
        
        # Cost/financial questions
        if any(word in question for word in ['cost', 'price', 'budget', 'expensive', 'cheap', 'money']):
            return 'financial'
        
        # Maintenance questions
        if any(word in question for word in ['maintain', 'clean', 'repair', 'service', 'check']):
            return 'maintenance'
        
        # Installation questions
        if any(word in question for word in ['install', 'setup', 'mount', 'connect', 'wire']):
            return 'installation'
        
        # Comparison questions
        if any(word in question for word in ['compare', 'difference', 'better', 'best', 'vs']):
            return 'comparison'
        
        return 'general'
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question"""
        # Solar-related terms
        solar_terms = ['solar', 'panel', 'battery', 'inverter', 'system', 'energy', 'power']
        
        # Technical terms
        tech_terms = ['watt', 'kwh', 'voltage', 'current', 'efficiency', 'capacity']
        
        # Component terms
        component_terms = ['monocrystalline', 'polycrystalline', 'lifepo4', 'lead-acid', 'hybrid']
        
        # Extract terms
        key_terms = []
        question_lower = question.lower()
        
        for term in solar_terms + tech_terms + component_terms:
            if term in question_lower:
                key_terms.append(term)
        
        return key_terms
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity"""
        # Simple heuristics
        word_count = len(question.split())
        technical_terms = len(self._extract_key_terms(question))
        
        if word_count < 5 and technical_terms < 2:
            return 'simple'
        elif word_count > 15 or technical_terms > 4:
            return 'complex'
        else:
            return 'medium'
    
    def _is_follow_up_question(self, question: str) -> bool:
        """Check if this is a follow-up question"""
        follow_up_indicators = ['also', 'additionally', 'furthermore', 'what about', 'and', 'but']
        return any(indicator in question.lower() for indicator in follow_up_indicators)
    
    def _preprocess_question(self, question: str) -> str:
        """Preprocess question for better matching"""
        # Remove common words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = question.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    
    def _retrieve_knowledge(self, question: str, analysis: Dict[str, Any]) -> List[KnowledgeEntry]:
        """Retrieve relevant knowledge entries"""
        try:
            relevant_entries = []
            
            # Search by key terms
            for term in analysis['key_terms']:
                for entry in self.knowledge_base:
                    if term in entry.title.lower() or term in entry.content.lower():
                        if entry not in relevant_entries:
                            relevant_entries.append(entry)
            
            # Search by question type
            question_type = analysis['question_type']
            for entry in self.knowledge_base:
                if question_type in entry.category.lower():
                    if entry not in relevant_entries:
                        relevant_entries.append(entry)
            
            # Sort by relevance
            relevant_entries.sort(key=lambda x: x.confidence, reverse=True)
            
            return relevant_entries[:5]  # Return top 5 most relevant
            
        except Exception as e:
            self.log_activity(f"Error retrieving knowledge: {e}", 'error')
            return []
    
    def _generate_answer(self, question: str, knowledge: List[KnowledgeEntry], 
                        context: Dict[str, Any], user_profile: Dict[str, Any]) -> QAAnswer:
        """Generate comprehensive answer"""
        try:
            # Use LLM if available
            if self.llm_manager and knowledge:
                answer = self._generate_llm_answer(question, knowledge, context, user_profile)
            else:
                answer = self._generate_rule_based_answer(question, knowledge, context)
            
            # Generate related questions
            related_questions = self._generate_related_questions(question, knowledge)
            
            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(question, answer)
            
            return QAAnswer(
                question=question,
                answer=answer['answer'],
                confidence=answer['confidence'],
                answer_type=answer['answer_type'],
                sources=answer['sources'],
                related_questions=related_questions,
                follow_up_suggestions=follow_up_suggestions,
                ai_explanation=answer.get('explanation', ''),
                context_used=context
            )
            
        except Exception as e:
            self.log_activity(f"Error generating answer: {e}", 'error')
            return self._create_fallback_answer(question)
    
    def _generate_llm_answer(self, question: str, knowledge: List[KnowledgeEntry], 
                           context: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using LLM"""
        try:
            llm = self.llm_manager.get_llm('reasoning')
            if not llm:
                return self._generate_rule_based_answer(question, knowledge, context)
            
            # Prepare context for LLM
            knowledge_context = "\n".join([f"- {entry.title}: {entry.content[:200]}..." for entry in knowledge[:3]])
            
            prompt = f"""
            Answer this solar energy question based on the provided knowledge:
            
            Question: {question}
            
            Knowledge Base:
            {knowledge_context}
            
            User Context: {context}
            User Profile: {user_profile}
            
            Provide a comprehensive, accurate answer with confidence level.
            """
            
            response = llm.invoke(prompt)
            
            return {
                'answer': response,
                'confidence': 0.85,
                'answer_type': 'llm_generated',
                'sources': [entry.source for entry in knowledge[:3]],
                'explanation': 'Generated using AI reasoning'
            }
            
        except Exception as e:
            self.log_activity(f"LLM answer generation failed: {e}", 'warning')
            return self._generate_rule_based_answer(question, knowledge, context)
    
    def _generate_rule_based_answer(self, question: str, knowledge: List[KnowledgeEntry], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using rule-based approach"""
        try:
            if not knowledge:
                return self._create_fallback_answer(question)
            
            # Use the most relevant knowledge entry
            best_entry = knowledge[0]
            
            # Create answer based on knowledge
            answer = f"Based on the available information: {best_entry.content}"
            
            return {
                'answer': answer,
                'confidence': best_entry.confidence,
                'answer_type': 'rule_based',
                'sources': [best_entry.source],
                'explanation': 'Generated using knowledge base matching'
            }
            
        except Exception as e:
            self.log_activity(f"Rule-based answer generation failed: {e}", 'error')
            return self._create_fallback_answer(question)
    
    def _generate_related_questions(self, question: str, knowledge: List[KnowledgeEntry]) -> List[str]:
        """Generate related questions"""
        related_questions = []
        
        # Generate based on knowledge categories
        categories = set(entry.category for entry in knowledge)
        
        for category in categories:
            if category == 'technical':
                related_questions.append("What are the technical specifications I should consider?")
            elif category == 'financial':
                related_questions.append("What is the cost breakdown for this system?")
            elif category == 'maintenance':
                related_questions.append("How do I maintain this system?")
        
        return related_questions[:3]  # Return top 3
    
    def _generate_follow_up_suggestions(self, question: str, answer: QAAnswer) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = []
        
        # Based on question type
        if 'cost' in question.lower():
            suggestions.append("Would you like to see a detailed cost breakdown?")
        
        if 'install' in question.lower():
            suggestions.append("Do you need installation guidance?")
        
        if 'maintain' in question.lower():
            suggestions.append("Would you like maintenance tips?")
        
        return suggestions[:3]  # Return top 3
    
    def _create_fallback_answer(self, question: str) -> QAAnswer:
        """Create fallback answer when all else fails"""
        return QAAnswer(
            question=question,
            answer="I apologize, but I don't have enough information to provide a comprehensive answer to your question. Please try rephrasing your question or contact our support team for assistance.",
            confidence=0.3,
            answer_type="fallback",
            sources=[],
            related_questions=[],
            follow_up_suggestions=["Try rephrasing your question", "Contact support for assistance"],
            ai_explanation="Fallback response due to insufficient information",
            context_used={}
        )
    
    def _update_conversation_history(self, question: str, answer: QAAnswer):
        """Update conversation history"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'question': question,
            'answer': answer.answer,
            'confidence': answer.confidence
        })
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _load_knowledge_base(self) -> List[KnowledgeEntry]:
        """Load knowledge base"""
        try:
            knowledge_entries = []
            
            # Load from CSV if available
            if os.path.exists('data/interim/cleaned/qa_knowledge.csv'):
                df = pd.read_csv('data/interim/cleaned/qa_knowledge.csv')
                for _, row in df.iterrows():
                    entry = KnowledgeEntry(
                        id=str(row.get('id', '')),
                        title=str(row.get('title', '')),
                        content=str(row.get('content', '')),
                        category=str(row.get('category', 'general')),
                        tags=row.get('tags', '').split(',') if row.get('tags') else [],
                        confidence=float(row.get('confidence', 0.8)),
                        source=str(row.get('source', 'knowledge_base')),
                        last_updated=datetime.now()
                    )
                    knowledge_entries.append(entry)
            else:
                # Create default knowledge entries
                knowledge_entries = self._create_default_knowledge()
            
            self.log_activity(f"Loaded {len(knowledge_entries)} knowledge entries")
            return knowledge_entries
            
        except Exception as e:
            self.log_activity(f"Error loading knowledge base: {e}", 'error')
            return self._create_default_knowledge()
    
    def _create_default_knowledge(self) -> List[KnowledgeEntry]:
        """Create default knowledge entries"""
        return [
            KnowledgeEntry(
                id="solar_basics_1",
                title="How Solar Panels Work",
                content="Solar panels convert sunlight into electricity using photovoltaic cells. When sunlight hits the cells, it creates an electric current that can power your home.",
                category="technical",
                tags=["solar", "panels", "electricity"],
                confidence=0.9,
                source="solar_basics",
                last_updated=datetime.now()
            ),
            KnowledgeEntry(
                id="battery_types_1",
                title="Battery Types for Solar Systems",
                content="LiFePO4 batteries are the best choice for solar systems due to their long lifespan, high efficiency, and safety. Lead-acid batteries are cheaper but have shorter lifespans.",
                category="technical",
                tags=["battery", "lifepo4", "lead-acid"],
                confidence=0.85,
                source="battery_guide",
                last_updated=datetime.now()
            ),
            KnowledgeEntry(
                id="maintenance_1",
                title="Solar System Maintenance",
                content="Regular maintenance includes cleaning panels monthly, checking connections quarterly, and monitoring battery levels weekly. Nigerian dust conditions require more frequent cleaning.",
                category="maintenance",
                tags=["maintenance", "cleaning", "monitoring"],
                confidence=0.8,
                source="maintenance_guide",
                last_updated=datetime.now()
            )
        ]
    
    def _load_answer_templates(self) -> Dict[str, str]:
        """Load answer templates"""
        return {
            'technical': "Based on technical specifications: {answer}",
            'financial': "From a cost perspective: {answer}",
            'maintenance': "For maintenance considerations: {answer}",
            'general': "{answer}"
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get Q&A Agent status"""
        return {
            'agent_name': self.agent_name,
            'status': self.status,
            'capabilities': [
                'Intelligent question answering',
                'Context-aware responses',
                'Knowledge base retrieval',
                'LLM-powered reasoning',
                'Conversation history tracking'
            ],
            'knowledge_base_size': len(self.knowledge_base),
            'conversation_history_size': len(self.conversation_history),
            'answer_templates': len(self.answer_templates)
        }