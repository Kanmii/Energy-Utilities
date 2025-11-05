"""
Chat Interface Agent - Multi-LLM Powered Conversational Interface
Handles conversational interface with advanced LLM integration using all 4 LLMs
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

# Import core infrastructure
try:
    from ..core.nlp_processor import NLPProcessor
    from ..core.llm_manager import StreamlinedLLMManager
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from nlp_processor import NLPProcessor
    from llm_manager import StreamlinedLLMManager

@dataclass
class ChatMessage:
    """Chat message structure"""
    user_id: str
    message: str
    timestamp: datetime
    intent: str
    entities: Dict[str, Any]
    context: Dict[str, Any]
    response: str

@dataclass
class ConversationState:
    """Conversation state management"""
    user_id: str
    current_intent: str
    collected_data: Dict[str, Any]
    missing_fields: List[str]
    conversation_history: List[ChatMessage]
    context: Dict[str, Any]

class ChatInterfaceAgent:
    """
    Multi-LLM Powered Chat Interface Agent
    Uses all 4 LLMs working together for intelligent conversation:
    - Groq Llama3: Fast reasoning and intent analysis
    - Groq Mixtral: Complex problem solving
    - HuggingFace: Open-source model diversity
    - Replicate: Advanced model capabilities
    - OpenRouter: Access to latest models
    """
    
    def __init__(self):
        self.agent_name = "ChatInterfaceAgent"
        self.status = "active"
        
        # Initialize LLM Manager with all 4 LLMs
        self.llm_manager = StreamlinedLLMManager()
        
        # Initialize NLP processor
        self.nlp_processor = NLPProcessor()
        
        # Conversation management
        self.conversations = {}
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        
        # Multi-LLM task assignment
        self.llm_tasks = {
            'intent_analysis': 'groq_llama3',      # Fast intent recognition
            'context_understanding': 'groq_mixtral', # Deep context analysis
            'response_generation': 'openrouter_claude', # High-quality responses
            'knowledge_retrieval': 'huggingface',   # Knowledge-based queries
            'creative_responses': 'replicate'       # Creative and engaging responses
        }
        
        # Required fields for solar system analysis
        self.required_fields = [
            'location', 'appliances', 'budget', 'backup_days', 'system_type'
        ]
        
        print(f"🤖 {self.agent_name} initialized with Multi-LLM System:")
        available_llms = self.llm_manager.get_available_providers()
        for llm in available_llms:
            print(f"   {llm}")
        print(f"   Total LLMs: {len(available_llms)}")
 
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Process user message and generate response
        """
        try:
            # Get or create conversation state
            if user_id not in self.conversations:
                self.conversations[user_id] = ConversationState(
                    user_id=user_id,
                    current_intent='greeting',
                    collected_data={},
                    missing_fields=self.required_fields.copy(),
                    conversation_history=[],
                    context={}
                )
            
            conversation = self.conversations[user_id]
            
            # Multi-LLM Processing Pipeline
            # Step 1: Fast intent analysis with Groq Llama3
            intent_result = await self._analyze_intent_with_llm(message)
            
            # Step 2: Deep context understanding with Groq Mixtral
            context_result = await self._understand_context_with_llm(message, conversation)
            
            # Step 3: Process message with NLP
            nlp_result = await self._process_with_nlp(message)
            
            # Combine LLM and NLP results
            combined_result = {
                **nlp_result,
                'intent_analysis': intent_result,
                'context_understanding': context_result
            }
            
            # Update conversation state
            self._update_conversation_state(conversation, message, combined_result)
            
            # Step 4: Generate intelligent response using best LLM for the task
            response = await self._generate_multi_llm_response(conversation, combined_result)
            
            # Store message in history
            chat_message = ChatMessage(
                user_id=user_id,
                message=message,
                timestamp=datetime.now(),
                intent=nlp_result['intent'],
                entities=nlp_result['entities'],
                context=nlp_result['context'],
                response=response
            )
            conversation.conversation_history.append(chat_message)
            
            # Check if we have enough data for system analysis
            analysis_ready = self._check_analysis_readiness(conversation)
            
            return {
                'success': True,
                'response': response,
                'intent': nlp_result['intent'],
                'entities': nlp_result['entities'],
                'conversation_state': {
                    'current_intent': conversation.current_intent,
                    'collected_data': conversation.collected_data,
                    'missing_fields': conversation.missing_fields,
                    'analysis_ready': analysis_ready
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': "I'm sorry, I didn't understand that. Could you please rephrase?",
                'timestamp': datetime.now().isoformat()
            }
 
    async def _process_with_nlp(self, message: str) -> Dict[str, Any]:
        """Process message with NLP"""
        try:
            # Use NLP processor to get comprehensive analysis
            nlp_result = self.nlp_processor.process_input(message)
            
            # Intent classification (use our local classifier as fallback)
            intent = self.intent_classifier.classify(message)
            
            # Entity extraction (use our local extractor as fallback)
            entities = self.entity_extractor.extract(message)
            
            # Use NLP processor results if available
            if nlp_result:
                intent = nlp_result.get('intent', intent)
                entities.update(nlp_result.get('entities', {}))
                sentiment = nlp_result.get('sentiment', 'neutral')
                context = nlp_result.get('keywords', {})
            else:
                sentiment = 'neutral'
                context = {}
            
            return {
                'intent': intent,
                'entities': entities,
                'sentiment': sentiment,
                'context': context
            }
            
        except Exception as e:
            print(f"NLP processing failed: {e}")
            return {
                'intent': 'general_question',
                'entities': {},
                'sentiment': 'neutral',
                'context': {}
            }
 
    def _update_conversation_state(self, conversation: ConversationState, message: str, nlp_result: Dict[str, Any]):
        """Update conversation state with new information"""
        # Update intent
        conversation.current_intent = nlp_result['intent']
        
        # Update collected data with extracted entities
        for key, value in nlp_result['entities'].items():
            if value:
                conversation.collected_data[key] = value
                if key in conversation.missing_fields:
                    conversation.missing_fields.remove(key)
        
        # Update context
        conversation.context.update(nlp_result['context'])
    
    async def _generate_response(self, conversation: ConversationState, nlp_result: Dict[str, Any]) -> str:
        """Generate appropriate response based on conversation state"""
        try:
            return await self.response_generator.generate_response(
                conversation, nlp_result
            )
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "I'm here to help you with your solar system needs. What would you like to know?"
    
    def _check_analysis_readiness(self, conversation: ConversationState) -> bool:
        """Check if we have enough data to proceed with system analysis"""
        required_fields = ['location', 'appliances', 'budget']
        return all(field in conversation.collected_data for field in required_fields)
    
    def get_conversation_state(self, user_id: str) -> Optional[ConversationState]:
        """Get conversation state for user"""
        return self.conversations.get(user_id)
    
    def reset_conversation(self, user_id: str) -> bool:
        """Reset conversation for user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            return True
        return False
    
    def get_collected_data(self, user_id: str) -> Dict[str, Any]:
        """Get collected data for user"""
        if user_id in self.conversations:
            return self.conversations[user_id].collected_data
        return {}

class IntentClassifier:
    """Intent classification for user messages"""
    
    def __init__(self):
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'quote_request': ['need', 'want', 'looking for', 'quote', 'price', 'cost', 'solar system'],
            'system_sizing': ['size', 'calculate', 'how much', 'power', 'capacity', 'sizing'],
            'appliance_question': ['appliance', 'device', 'equipment', 'fan', 'tv', 'light', 'fridge'],
            'location_question': ['location', 'area', 'city', 'state', 'address', 'where'],
            'budget_question': ['budget', 'cost', 'price', 'afford', 'money', 'expensive'],
            'backup_question': ['backup', 'battery', 'power outage', 'blackout', 'autonomy'],
            'general_question': ['what', 'how', 'why', 'when', 'where', 'explain', 'tell me'],
            'goodbye': ['bye', 'goodbye', 'thanks', 'thank you', 'see you']
        }
    
    def classify(self, message: str) -> str:
        """Classify user intent"""
        message_lower = message.lower()
        
        # Check for exact matches first
        for intent, keywords in self.intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        # Default to general question
        return 'general_question'

class EntityExtractor:
    """Entity extraction from user messages"""
    
    def __init__(self):
        self.location_patterns = [
            r'in\s+([A-Za-z\s,]+?)(?:\s|$)',
            r'at\s+([A-Za-z\s,]+?)(?:\s|$)',
            r'from\s+([A-Za-z\s,]+?)(?:\s|$)',
            r'location[:\s]+([A-Za-z\s,]+?)(?:\s|$)',
            r'address[:\s]+([A-Za-z\s,]+?)(?:\s|$)'
        ]
        
        self.budget_patterns = [
            r'budget[:\s]+([0-9,]+)',
            r'around\s+([0-9,]+)',
            r'up\s+to\s+([0-9,]+)',
            r'about\s+([0-9,]+)',
            r'approximately\s+([0-9,]+)',
            r'([0-9,]+)\s+naira',
            r'([0-9,]+)\s+₦'
        ]
        
        self.appliance_patterns = [
            r'(\d+)\s+([A-Za-z\s]+)\s+fan',
            r'(\d+)\s+([A-Za-z\s]+)\s+tv',
            r'(\d+)\s+([A-Za-z\s]+)\s+light',
            r'(\d+)\s+([A-Za-z\s]+)\s+fridge',
            r'(\d+)\s+([A-Za-z\s]+)\s+air\s+conditioner',
            r'(\d+)\s+([A-Za-z\s]+)\s+ac'
        ]
        
        self.backup_patterns = [
            r'(\d+)\s+days?\s+backup',
            r'(\d+)\s+hours?\s+backup',
            r'backup\s+for\s+(\d+)\s+days?',
            r'backup\s+for\s+(\d+)\s+hours?'
        ]
 
    def extract(self, message: str) -> Dict[str, Any]:
        """Extract entities from message"""
        entities = {}
        
        # Extract location
        for pattern in self.location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 2:  # Filter out very short matches
                    entities['location'] = location
                    break
        
        # Extract budget
        for pattern in self.budget_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                budget = match.group(1).replace(',', '')
                if budget.isdigit():
                    entities['budget'] = int(budget)
                    break
        
        # Extract appliances
        appliances = []
        for pattern in self.appliance_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                quantity = int(match[0])
                appliance_type = match[1].strip()
                appliances.append({
                    'type': appliance_type,
                    'quantity': quantity
                })
        
        if appliances:
            entities['appliances'] = appliances
        
        # Extract backup days
        for pattern in self.backup_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                backup = int(match.group(1))
                entities['backup_days'] = backup
                break
        
        return entities

class ResponseGenerator:
    """Generate appropriate responses based on conversation state"""
    
    def __init__(self):
        self.response_templates = {
            'greeting': [
                "Hello! I'm here to help you with your solar system needs. What would you like to know?",
                "Hi there! I can help you design a solar system for your home. What's your location?",
                "Welcome! I'm your solar system assistant. Where are you located?"
            ],
            'quote_request': [
                "Great! I'd be happy to help you get a quote for a solar system. What's your location?",
                "I can help you with a solar system quote. First, where are you located?",
                "Let's get you a solar system quote! What's your address or city?"
            ],
            'system_sizing': [
                "I can help you calculate the right size for your solar system. What appliances do you want to power?",
                "Let's size your solar system! What devices do you need to power?",
                "I'll help you determine the right system size. What appliances will you be using?"
            ],
            'appliance_question': [
                "What appliances do you want to power with solar? For example, fans, TVs, lights, etc.",
                "Tell me about your appliances. How many fans, TVs, lights do you have?",
                "I need to know your appliances to size the system. What devices will you be using?"
            ],
            'location_question': [
                "What's your location? I need this to calculate solar potential and find local installers.",
                "Where are you located? This helps me determine the best system for your area.",
                "Please provide your location so I can customize the solar system for your area."
            ],
            'budget_question': [
                "What's your budget for the solar system? This helps me recommend the best options.",
                "How much are you looking to spend on your solar system?",
                "What's your budget range? I can suggest systems within your price range."
            ],
            'backup_question': [
                "How many days of backup power do you need? This affects battery sizing.",
                "What's your backup power requirement? How many days should the system last without sun?",
                "For backup power, how many days do you want the system to run without sunlight?"
            ],
            'general_question': [
                "I'm here to help with your solar system needs. What would you like to know?",
                "I can help you with solar system sizing, quotes, and recommendations. What do you need?",
                "Ask me anything about solar systems! I can help with sizing, costs, and installation."
            ],
            'goodbye': [
                "Thank you for using our solar system assistant! Have a great day!",
                "Goodbye! Feel free to come back anytime for solar system help.",
                "Thanks for chatting! I hope I helped with your solar system needs."
            ]
        }
 
    async def generate_response(self, conversation: ConversationState, nlp_result: Dict[str, Any]) -> str:
        """Generate appropriate response"""
        intent = nlp_result['intent']
        entities = nlp_result['entities']
        
        # Check if we have enough data
        if len(conversation.missing_fields) == 0:
            return "Perfect! I have all the information I need. Let me analyze your requirements and provide you with a solar system recommendation."
        
        # Generate response based on intent and missing fields
        if intent in self.response_templates:
            response = self.response_templates[intent][0]
        else:
            response = self.response_templates['general_question'][0]
        
        # Add specific questions for missing fields
        if 'location' in conversation.missing_fields:
            response += " What's your location?"
        elif 'appliances' in conversation.missing_fields:
            response += " What appliances do you want to power?"
        elif 'budget' in conversation.missing_fields:
            response += " What's your budget for the solar system?"
        elif 'backup_days' in conversation.missing_fields:
            response += " How many days of backup power do you need?"
        
        return response
    
    async def _analyze_intent_with_llm(self, message: str) -> Dict[str, Any]:
        """Analyze user intent using Groq Llama3 for fast processing"""
        try:
            llm = self.llm_manager.get_llm(self.llm_tasks['intent_analysis'])
            if not llm:
                return {'intent': 'unknown', 'confidence': 0.0}
            
            prompt = f"""Analyze the user's intent in this solar energy conversation:
            
User message: "{message}"

Look for specific appliances mentioned (fridge, AC, TV, lights, laptop, freezer, etc.) and budget amounts.

Classify the intent as one of:
- system_sizing: asking about solar system size/requirements OR listing specific appliances
- appliance_info: asking about appliance power consumption only
- location_query: asking about solar potential in their area
- budget_inquiry: asking about costs and pricing only
- education: asking how solar technology works
- general: general conversation or greeting

If appliances are mentioned (like "fridge", "AC", "TV", "lights"), classify as system_sizing.

Respond with JSON format:
{{"intent": "category", "confidence": 0.0-1.0, "reasoning": "brief explanation", "appliances_mentioned": true/false, "budget_mentioned": true/false}}"""

            response = llm.invoke(prompt)
            
            # Parse JSON response
            try:
                import json
                result = json.loads(response.content if hasattr(response, 'content') else str(response))
                return result
            except:
                return {
                    'intent': 'general',
                    'confidence': 0.5,
                    'reasoning': 'Failed to parse LLM response'
                }
                
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return {'intent': 'unknown', 'confidence': 0.0}
    
    async def _understand_context_with_llm(self, message: str, conversation: ConversationState) -> Dict[str, Any]:
        """Deep context understanding using Groq Mixtral"""
        try:
            llm = self.llm_manager.get_llm(self.llm_tasks['context_understanding'])
            if not llm:
                return {'context_score': 0.0, 'key_entities': []}
            
            # Build conversation history for context
            history = ""
            for msg in conversation.conversation_history[-3:]:  # Last 3 messages
                history += f"User: {msg.message}\nAssistant: {msg.response}\n"
            
            prompt = f"""Analyze the context and extract key information from this solar energy conversation:

Conversation History:
{history}

Current User Message: "{message}"

Current collected data: {conversation.collected_data}
Missing fields: {conversation.missing_fields}

Extract and analyze:
1. Key entities (locations, appliances, numbers, etc.)
2. Context continuity score (0.0-1.0)
3. User's expertise level (beginner/intermediate/advanced)
4. Urgency level (low/medium/high)

Respond with JSON:
{{"key_entities": [], "context_score": 0.0, "expertise_level": "beginner", "urgency": "medium", "summary": "brief context summary"}}"""

            response = llm.invoke(prompt)
            
            try:
                import json
                result = json.loads(response.content if hasattr(response, 'content') else str(response))
                return result
            except:
                return {
                    'key_entities': [],
                    'context_score': 0.5,
                    'expertise_level': 'beginner',
                    'urgency': 'medium'
                }
                
        except Exception as e:
            print(f"Context understanding error: {e}")
            return {'context_score': 0.0, 'key_entities': []}
    
    async def _generate_multi_llm_response(self, conversation: ConversationState, combined_result: Dict[str, Any]) -> str:
        """Generate response using the most appropriate LLM based on intent and context"""
        try:
            intent = combined_result.get('intent_analysis', {}).get('intent', 'general')
            context = combined_result.get('context_understanding', {})
            expertise_level = context.get('expertise_level', 'beginner')
            
            # Choose best LLM for the task
            if intent in ['education', 'general']:
                llm_key = self.llm_tasks['creative_responses']  # Replicate for engaging explanations
            elif intent in ['system_sizing', 'budget_inquiry']:
                llm_key = self.llm_tasks['response_generation']  # OpenRouter for detailed responses
            elif intent in ['appliance_info', 'location_query']:
                llm_key = self.llm_tasks['knowledge_retrieval']  # HuggingFace for factual info
            else:
                llm_key = self.llm_tasks['intent_analysis']  # Groq Llama3 for general responses
            
            llm = self.llm_manager.get_llm(llm_key)
            if not llm:
                # Fallback to any available LLM
                available_llms = self.llm_manager.get_available_providers()
                if available_llms:
                    llm = self.llm_manager.get_llm(available_llms[0])
                else:
                    return "I'm having trouble accessing my AI models. Please try again."
            
            # Build comprehensive prompt with human touch
            prompt = f"""You're a friendly solar energy expert who genuinely cares about helping Nigerian families achieve energy independence!

**About this conversation:**
- User Intent: {intent}
- Their Experience Level: {expertise_level}
- Context Understanding: {context.get('context_score', 0.5)}

**What they just said:** "{conversation.conversation_history[-1].message if conversation.conversation_history else 'Hello'}"

**What we know about them:** {conversation.collected_data}
**What we still need to help them:** {conversation.missing_fields}

**Your personality:**
- Warm and encouraging (like a knowledgeable friend)
- Use simple, everyday language
- Show genuine excitement about solar energy
- Be empathetic about their concerns (especially budget!)
- Use "you" and "your" to make it personal
- Include encouraging phrases and emojis where appropriate
- Share relatable Nigerian examples

**Your response should:**
- Match their {expertise_level} level (don't overwhelm beginners!)
- If you need more info, ask in a friendly, curious way
- Give specific, actionable advice they can use
- Make them feel confident about their solar journey
- Use Nigerian context (mention cities, local challenges, etc.)

Remember: You're not just providing information - you're helping them achieve their dream of reliable, affordable power!"""

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response
            response_text = response_text.strip()
            if len(response_text) > 1000:
                response_text = response_text[:1000] + "..."
            
            return response_text
            
        except Exception as e:
            print(f"Multi-LLM response generation error: {e}")
            return "I'm here to help with your solar energy questions. What would you like to know?"

# Example usage
async def main():
    """Example usage of ChatInterfaceAgent"""
    chat_agent = ChatInterfaceAgent()
    
    # Simulate conversation
    user_id = "test_user"
    
    messages = [
        "Hello, I need a solar system",
        "I'm in Lagos, Nigeria",
        "I have 3 fans and 2 TVs",
        "My budget is 500,000 naira",
        "I need 2 days backup"
    ]
    
    for message in messages:
        result = await chat_agent.process_message(user_id, message)
        print(f"User: {message}")
        print(f"Bot: {result['response']}")
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print(f"Analysis Ready: {result['conversation_state']['analysis_ready']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
