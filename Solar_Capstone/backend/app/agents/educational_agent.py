#!/usr/bin/env python3
"""
Educational Agent - Multi-LLM Powered Solar Learning Assistant
Advanced educational agent using all 4 LLMs for personalized learning:
- Groq Llama3: Quick concept explanations
- Groq Mixtral: Complex technical analysis
- HuggingFace: Educational content generation
- Replicate: Creative learning materials
- OpenRouter: Advanced pedagogical responses
"""
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import LLM Manager
try:
    from ..core.llm_manager import StreamlinedLLMManager
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from llm_manager import StreamlinedLLMManager

@dataclass
class EducationalContent:
    """Educational content structure"""
    title: str
    content: str
    category: str
    difficulty_level: str
    estimated_read_time: int  # minutes
    tags: List[str]
    related_topics: List[str]
    # Enhanced features
    learning_objectives: List[str] = None  # Learning objectives
    key_concepts: List[str] = None  # Key concepts
    examples: List[str] = None  # Examples
    visual_aids: List[str] = None  # Visual aids
    next_steps: List[str] = None  # Next steps
    ai_explanation: str = ""  # LLM-generated explanation
    personalized_notes: str = ""  # User-specific insights

@dataclass
class ExplanationStep:
    """Step in an explanation process"""
    step_number: int
    title: str
    description: str
    formula: Optional[str]
    example: Optional[str]
    visual_aid: Optional[str]

@dataclass
class UserGuidance:
    """User guidance and recommendations"""
    guidance_type: str  # "beginner", "intermediate", "advanced"
    recommendations: List[str]
    warnings: List[str]
    tips: List[str]
    next_steps: List[str]

@dataclass
class EducationalResponse:
    """Complete educational response"""
    user_guidance: UserGuidance
    explanations: List[ExplanationStep]
    educational_content: List[EducationalContent]
    faq_suggestions: List[str]
    confidence: float

class EducationalAgent:
    """
    Multi-LLM Powered Educational Agent for Solar Learning
    
    Uses all 4 LLMs strategically:
    - Groq Llama3: Fast concept explanations and quick answers
    - Groq Mixtral: Complex technical analysis and detailed explanations
    - HuggingFace: Educational content generation and knowledge retrieval
    - Replicate: Creative learning materials and analogies
    - OpenRouter: Advanced pedagogical responses and personalized learning
    """
    
    def __init__(self):
        """Initialize the Multi-LLM Educational Agent"""
        self.agent_name = "EducationalAgent"
        self.version = "2.0.0"
        
        # Initialize LLM Manager with all 4 LLMs
        self.llm_manager = StreamlinedLLMManager()
        
        # Multi-LLM task assignment for education
        self.llm_tasks = {
            'quick_explanations': 'groq_llama3',        # Fast concept explanations
            'technical_analysis': 'groq_mixtral',       # Complex technical details
            'content_generation': 'huggingface',        # Educational content creation
            'creative_learning': 'replicate',           # Analogies and creative explanations
            'personalized_teaching': 'openrouter_claude' # Advanced pedagogical responses
        }
        
        # Educational content database
        self.educational_content = self._load_educational_content()
        
        # Explanation templates
        self.explanation_templates = self._load_explanation_templates()
        
        # User guidance templates
        self.guidance_templates = self._load_guidance_templates()
        
        # FAQ database
        self.faq_database = self._load_faq_database()
        
        # Learning level adaptation
        self.learning_levels = {
            'beginner': {
                'primary_llm': 'groq_llama3',
                'secondary_llm': 'replicate',
                'complexity': 'simple',
                'use_analogies': True
            },
            'intermediate': {
                'primary_llm': 'groq_mixtral',
                'secondary_llm': 'huggingface',
                'complexity': 'moderate',
                'use_analogies': True
            },
            'advanced': {
                'primary_llm': 'openrouter_claude',
                'secondary_llm': 'groq_mixtral',
                'complexity': 'detailed',
                'use_analogies': False
            }
        }
        
        # RAG integration disabled
        self.rag_kb = None
        self.enhanced_agent = None

        print(f"🎓 {self.agent_name} v{self.version} initialized with Multi-LLM System:")
        available_llms = self.llm_manager.get_available_providers()
        for llm in available_llms:
            print(f"   {llm}")
        print(f"   Educational LLMs: {len(available_llms)}")
        print(f"   Learning Levels: {list(self.learning_levels.keys())}")
    
    def _load_educational_content(self) -> Dict[str, List[EducationalContent]]:
        """Load educational content database"""
        return {
            "solar_basics": [
                EducationalContent(
                    title="Understanding Solar Energy",
                    content="Solar energy is the conversion of sunlight into electricity using photovoltaic (PV) cells. In Nigeria, we receive abundant sunlight year-round, making solar power an excellent renewable energy source.",
                    category="basics",
                    difficulty_level="beginner",
                    estimated_read_time=5,
                    tags=["solar", "energy", "basics"],
                    related_topics=["photovoltaic", "renewable energy", "sustainability"]
                ),
                EducationalContent(
                    title="Solar Panel Types and Efficiency",
                    content="There are three main types of solar panels: Monocrystalline (most efficient, 15-22%), Polycrystalline (moderate efficiency, 13-16%), and Thin-film (flexible, 10-13%). For Nigerian conditions, monocrystalline panels are recommended.",
                    category="components",
                    difficulty_level="intermediate",
                    estimated_read_time=8,
                    tags=["solar panels", "efficiency", "types"],
                    related_topics=["monocrystalline", "polycrystalline", "thin-film"]
                )
            ],
            "system_design": [
                EducationalContent(
                    title="System Sizing Fundamentals",
                    content="Solar system sizing involves calculating daily energy consumption, determining peak sun hours for your location, and accounting for system losses. The formula is: System Size (kW) = Daily Energy (kWh) ÷ Peak Sun Hours ÷ System Efficiency.",
                    category="design",
                    difficulty_level="intermediate",
                    estimated_read_time=12,
                    tags=["sizing", "calculations", "design"],
                    related_topics=["energy consumption", "peak sun hours", "system efficiency"]
                )
            ],
            "maintenance": [
                EducationalContent(
                    title="Solar System Maintenance",
                    content="Regular maintenance ensures optimal performance: Clean panels monthly, check connections quarterly, monitor battery levels weekly, and inspect mounting hardware annually. Nigerian dust and humidity require more frequent cleaning.",
                    category="maintenance",
                    difficulty_level="beginner",
                    estimated_read_time=6,
                    tags=["maintenance", "cleaning", "monitoring"],
                    related_topics=["panel cleaning", "battery maintenance", "system monitoring"]
                )
            ]
        }
    
    def _load_explanation_templates(self) -> Dict[str, List[ExplanationStep]]:
        """Load explanation templates for different processes"""
        return {
            "energy_calculation": [
                ExplanationStep(
                    step_number=1,
                    title="Identify Appliances",
                    description="List all electrical appliances you want to power with solar energy",
                    formula=None,
                    example="Refrigerator (200W), TV (100W), Lights (50W × 10)",
                    visual_aid="appliance_list"
                ),
                ExplanationStep(
                    step_number=2,
                    title="Calculate Daily Usage",
                    description="Multiply each appliance's power by its daily usage hours",
                    formula="Daily Energy (kWh) = Power (W) × Hours × Quantity ÷ 1000",
                    example="Refrigerator: 200W × 24h × 1 = 4.8 kWh/day",
                    visual_aid="energy_calculation"
                )
            ],
            "system_sizing": [
                ExplanationStep(
                    step_number=1,
                    title="Determine Peak Sun Hours",
                    description="Find the average peak sun hours for your location in Nigeria",
                    formula=None,
                    example="Lagos: 5.5 hours, Abuja: 6.2 hours, Kano: 6.8 hours",
                    visual_aid="sun_hours_map"
                ),
                ExplanationStep(
                    step_number=2,
                    title="Calculate Panel Power",
                    description="Determine the total solar panel power needed",
                    formula="Panel Power (W) = Daily Energy (kWh) ÷ Peak Sun Hours",
                    example="5.65 kWh ÷ 5.5 hours = 1.03 kW (1030W)",
                    visual_aid="panel_sizing"
                )
            ]
        }
    
    def _load_guidance_templates(self) -> Dict[str, UserGuidance]:
        """Load user guidance templates"""
        return {
            "beginner": UserGuidance(
                guidance_type="beginner",
                recommendations=[
                    "Start with a small system to understand solar basics",
                    "Choose reputable brands with good warranties",
                    "Consider professional installation for your first system",
                    "Learn about basic maintenance and monitoring"
                ],
                warnings=[
                    "Avoid very cheap components - they may not last",
                    "Don't oversize your system initially",
                    "Ensure proper ventilation for batteries",
                    "Check local regulations and permits"
                ],
                tips=[
                    "Monitor your energy consumption for 1-2 months before sizing",
                    "Consider future expansion when designing your system",
                    "Keep receipts and warranties for all components",
                    "Join local solar energy communities for support"
                ],
                next_steps=[
                    "Calculate your current energy consumption",
                    "Research local solar installers",
                    "Get quotes from multiple suppliers",
                    "Consider financing options"
                ]
            ),
            "intermediate": UserGuidance(
                guidance_type="intermediate",
                recommendations=[
                    "Consider hybrid systems with grid-tie capability",
                    "Invest in quality monitoring systems",
                    "Plan for system expansion and upgrades",
                    "Learn about advanced battery management"
                ],
                warnings=[
                    "Ensure proper system grounding and protection",
                    "Don't mix different battery chemistries",
                    "Consider seasonal variations in energy production",
                    "Plan for maintenance access to all components"
                ],
                tips=[
                    "Use energy monitoring to optimize consumption",
                    "Consider time-of-use rates if available",
                    "Learn about net metering policies",
                    "Invest in quality surge protection"
                ],
                next_steps=[
                    "Design system with expansion in mind",
                    "Research advanced monitoring solutions",
                    "Consider backup power strategies",
                    "Learn about grid-tie regulations"
                ]
            )
        }
    
    def _load_faq_database(self) -> Dict[str, List[str]]:
        """Load FAQ database"""
        return {
            "general": [
                "How long do solar panels last in Nigeria?",
                "What is the payback period for solar systems?",
                "Do solar panels work during rainy season?",
                "How much maintenance do solar systems require?",
                "Can I install solar panels myself?"
            ],
            "technical": [
                "What size inverter do I need?",
                "How do I calculate battery capacity?",
                "What is the difference between grid-tie and off-grid?",
                "How do I size my solar panel array?",
                "What is peak sun hours and why is it important?"
            ],
            "financial": [
                "What financing options are available?",
                "Are there government incentives for solar?",
                "How do I calculate ROI on solar investment?",
                "What is the cost per kWh for solar?",
                "How do I compare quotes from different installers?"
            ],
            "maintenance": [
                "How often should I clean solar panels?",
                "What maintenance do batteries require?",
                "How do I monitor system performance?",
                "What should I do if my system stops working?",
                "How do I troubleshoot common problems?"
            ]
        }
    
    def provide_user_guidance(self, user_profile: Dict[str, Any]) -> UserGuidance:
        """Provide personalized user guidance based on profile"""
        try:
            # Determine user level based on profile
            experience_level = user_profile.get("experience_level", "beginner")
            system_size = user_profile.get("system_size", "small")
            budget_range = user_profile.get("budget_range", "medium")
            
            # Select appropriate guidance template
            if experience_level in self.guidance_templates:
                base_guidance = self.guidance_templates[experience_level]
            else:
                base_guidance = self.guidance_templates["beginner"]
            
            # Customize guidance based on profile
            customized_guidance = UserGuidance(
                guidance_type=experience_level,
                recommendations=base_guidance.recommendations.copy(),
                warnings=base_guidance.warnings.copy(),
                tips=base_guidance.tips.copy(),
                next_steps=base_guidance.next_steps.copy()
            )
            
            # Add system-specific recommendations
            if system_size == "large":
                customized_guidance.recommendations.append(
                    "Consider professional system design for large installations"
                )
                customized_guidance.warnings.append(
                    "Large systems require more complex maintenance and monitoring"
                )
            
            if budget_range == "budget":
                customized_guidance.tips.append(
                    "Consider phased installation to spread costs over time"
                )
                customized_guidance.warnings.append(
                    "Don't compromise on safety equipment to save costs"
                )
            
            return customized_guidance
            
        except Exception as e:
            print(f"WARNING: Error providing user guidance: {e}")
            return self.guidance_templates["beginner"]
    
    def explain_calculation_process(self, process_type: str, data: Dict[str, Any]) -> List[ExplanationStep]:
        """Provide step-by-step explanation of calculation processes"""
        try:
            if process_type in self.explanation_templates:
                base_steps = self.explanation_templates[process_type]
            else:
                return []
            
            # Customize steps with actual data
            customized_steps = []
            for step in base_steps:
                customized_step = ExplanationStep(
                    step_number=step.step_number,
                    title=step.title,
                    description=step.description,
                    formula=step.formula,
                    example=step.example,
                    visual_aid=step.visual_aid
                )
                
                # Add real examples if data is available
                if process_type == "energy_calculation" and "appliances" in data:
                    if step.step_number == 1:
                        appliance_list = ", ".join([app["name"] for app in data["appliances"]])
                        customized_step.example = f"Appliances: {appliance_list}"
                
                customized_steps.append(customized_step)
            
            return customized_steps
            
        except Exception as e:
            print(f"WARNING: Error explaining calculation process: {e}")
            return []
    
    def get_educational_content(self, topics: List[str], difficulty: str = "beginner") -> List[EducationalContent]:
        """Get educational content for specific topics"""
        try:
            relevant_content = []
            
            for topic in topics:
                if topic in self.educational_content:
                    for content in self.educational_content[topic]:
                        if content.difficulty_level == difficulty or difficulty == "all":
                            relevant_content.append(content)
            
            # Sort by estimated read time
            relevant_content.sort(key=lambda x: x.estimated_read_time)
            
            return relevant_content[:5]  # Return top 5 most relevant
            
        except Exception as e:
            print(f"WARNING: Error getting educational content: {e}")
            return []
    
    def suggest_faqs(self, context: str, user_level: str = "beginner") -> List[str]:
        """Suggest relevant FAQs based on context"""
        try:
            suggested_faqs = []
            
            # Add general FAQs
            suggested_faqs.extend(self.faq_database["general"][:3])
            
            # Add context-specific FAQs
            if "technical" in context.lower():
                suggested_faqs.extend(self.faq_database["technical"][:2])
            
            if "financial" in context.lower():
                suggested_faqs.extend(self.faq_database["financial"][:2])
            
            if "maintenance" in context.lower():
                suggested_faqs.extend(self.faq_database["maintenance"][:2])
            
            # Remove duplicates and limit to 5
            suggested_faqs = list(set(suggested_faqs))[:5]
            
            return suggested_faqs
            
        except Exception as e:
            print(f"WARNING: Error suggesting FAQs: {e}")
            return self.faq_database["general"][:3]
    
    def generate_educational_response(self, 
                                    user_profile: Dict[str, Any],
                                    system_data: Dict[str, Any],
                                    context: str = "general") -> EducationalResponse:
        """Generate comprehensive educational response"""
        try:
            # Get user guidance
            user_guidance = self.provide_user_guidance(user_profile)
            
            # Get explanations for system calculations
            explanations = []
            if "energy_calculation" in context:
                explanations.extend(self.explain_calculation_process("energy_calculation", system_data))
            if "system_sizing" in context:
                explanations.extend(self.explain_calculation_process("system_sizing", system_data))
            
            # Get educational content
            topics = ["solar_basics", "system_design"]
            if "maintenance" in context:
                topics.append("maintenance")
            
            educational_content = self.get_educational_content(
                topics, 
                user_profile.get("experience_level", "beginner")
            )
            
            # Get FAQ suggestions
            faq_suggestions = self.suggest_faqs(context, user_profile.get("experience_level", "beginner"))
            
            # Calculate confidence
            confidence = 0.8  # Base confidence
            if len(explanations) > 0:
                confidence += 0.1
            if len(educational_content) > 0:
                confidence += 0.1
            
            return EducationalResponse(
                user_guidance=user_guidance,
                explanations=explanations,
                educational_content=educational_content,
                faq_suggestions=faq_suggestions,
                confidence=min(confidence, 1.0)
            )
            
        except Exception as e:
            print(f"WARNING: Error generating educational response: {e}")
            # Return basic response
            return EducationalResponse(
                user_guidance=self.guidance_templates["beginner"],
                explanations=[],
                educational_content=[],
                faq_suggestions=self.faq_database["general"][:3],
                confidence=0.5
            )
    
    def get_system_explanation(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed explanation of system recommendations"""
        try:
            explanation = {
                "system_overview": {
                    "title": "Your Solar System Explained",
                    "description": "Here's how your solar system works and why we recommend these components"
                },
                "component_explanations": {},
                "calculation_summary": {},
                "maintenance_guide": {},
                "troubleshooting_tips": []
            }
            
            # Add component explanations
            if "sizing_result" in system_data:
                sizing = system_data["sizing_result"]
                explanation["component_explanations"]["solar_panels"] = {
                    "purpose": "Convert sunlight into electricity",
                    "sizing_reason": f"We recommend {sizing.get('panel_power_watts', 0):.0f}W of panels to meet your daily energy needs",
                    "efficiency": "Modern panels are 15-22% efficient at converting sunlight to electricity"
                }
                
                explanation["component_explanations"]["batteries"] = {
                    "purpose": "Store excess energy for use when the sun isn't shining",
                    "sizing_reason": f"We recommend {sizing.get('battery_capacity_kwh', 0):.1f} kWh of battery storage for {sizing.get('autonomy_days', 0):.0f} days of backup power",
                    "chemistry": f"{sizing.get('battery_chemistry', 'LiFePO4')} batteries offer the best performance and lifespan"
                }
            
            # Add calculation summary
            explanation["calculation_summary"] = {
                "daily_energy": f"Your daily energy consumption is {system_data.get('daily_energy', 0):.2f} kWh",
                "system_efficiency": f"System efficiency is {sizing.get('system_efficiency', 0):.1%}",
                "backup_hours": f"Your system can provide {sizing.get('backup_hours', 0):.1f} hours of backup power"
            }
            
            # Add maintenance guide
            explanation["maintenance_guide"] = {
                "daily": ["Check system status on monitoring app"],
                "weekly": ["Check battery charge levels", "Review energy production"],
                "monthly": ["Clean solar panels", "Check connections"],
                "quarterly": ["Professional system inspection", "Battery maintenance check"]
            }
            
            # Add troubleshooting tips
            explanation["troubleshooting_tips"] = [
                "If system stops working, check all connections and fuses",
                "Low battery levels may indicate insufficient solar production",
                "Reduced energy production may require panel cleaning",
                "Contact your installer for complex technical issues"
            ]
            
            return explanation
            
        except Exception as e:
            print(f"WARNING: Error generating system explanation: {e}")
            return {
                "system_overview": {"title": "System Explanation", "description": "Error generating explanation"},
                "component_explanations": {},
                "calculation_summary": {},
                "maintenance_guide": {},
                "troubleshooting_tips": []
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get Educational Agent status"""
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "status": "ready",
            "capabilities": [
                "User guidance and recommendations",
                "Transparent calculation explanations",
                "Educational content delivery",
                "FAQ suggestions",
                "System explanations"
            ],
            "content_database": {
                "educational_content": sum(len(content) for content in self.educational_content.values()),
                "explanation_templates": len(self.explanation_templates),
                "guidance_templates": len(self.guidance_templates),
                "faq_database": sum(len(faqs) for faqs in self.faq_database.values())
            }
        }
    
    def get_educational_content_rag(self, query: str, user_level: str = "beginner") -> Dict[str, Any]:
        """Get educational content using RAG knowledge base (fallback to original method)"""
        try:
            if self.enhanced_agent:
                return self.enhanced_agent.get_educational_content(query, user_level)
            else:
                # Fallback to original method
                return self.get_educational_content([query], user_level)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def explain_concept_with_multi_llm(self, concept: str, user_level: str = "beginner") -> Dict[str, Any]:
        """Explain solar concept using multiple LLMs for comprehensive understanding"""
        try:
            level_config = self.learning_levels.get(user_level, self.learning_levels['beginner'])
            
            # Step 1: Quick explanation with primary LLM
            quick_explanation = await self._get_quick_explanation(concept, level_config)
            
            # Step 2: Technical details with secondary LLM (if needed)
            technical_details = None
            if user_level in ['intermediate', 'advanced']:
                technical_details = await self._get_technical_details(concept, level_config)
            
            # Step 3: Creative analogies (for beginners and intermediates)
            analogies = None
            if level_config['use_analogies']:
                analogies = await self._get_creative_analogies(concept)
            
            # Step 4: Personalized learning path
            learning_path = await self._get_personalized_learning_path(concept, user_level)
            
            return {
                'success': True,
                'concept': concept,
                'user_level': user_level,
                'quick_explanation': quick_explanation,
                'technical_details': technical_details,
                'analogies': analogies,
                'learning_path': learning_path,
                'confidence': 0.9
            }
            
        except Exception as e:
            print(f"Multi-LLM concept explanation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_explanation': f"I can help explain {concept}. What specific aspect would you like to understand?"
            }
    
    async def _get_quick_explanation(self, concept: str, level_config: Dict[str, Any]) -> str:
        """Get quick explanation using primary LLM"""
        try:
            llm = self.llm_manager.get_llm(level_config['primary_llm'])
            if not llm:
                return f"Quick explanation of {concept} is not available right now."
            
            complexity = level_config['complexity']
            prompt = f"""Explain the solar energy concept "{concept}" in a {complexity} way.

Guidelines:
- Keep it concise (2-3 sentences)
- Use {complexity} language
- Focus on practical understanding
- Include Nigerian context if relevant

Concept to explain: {concept}"""

            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Quick explanation error: {e}")
            return f"I can explain {concept} - it's an important solar energy concept."
    
    async def _get_technical_details(self, concept: str, level_config: Dict[str, Any]) -> str:
        """Get technical details using secondary LLM"""
        try:
            llm = self.llm_manager.get_llm(level_config['secondary_llm'])
            if not llm:
                return None
            
            prompt = f"""Provide technical details about the solar energy concept "{concept}".

Include:
- Technical specifications
- How it works mechanically/electrically
- Efficiency considerations
- Common problems and solutions
- Industry standards and best practices

Make it detailed but accessible for someone with technical interest.

Concept: {concept}"""

            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Technical details error: {e}")
            return None
    
    async def _get_creative_analogies(self, concept: str) -> List[str]:
        """Get creative analogies using Replicate LLM"""
        try:
            llm = self.llm_manager.get_llm(self.llm_tasks['creative_learning'])
            if not llm:
                return []
            
            prompt = f"""Create 2-3 simple, relatable analogies to explain the solar energy concept "{concept}".

Use everyday objects and situations that people in Nigeria would understand.
Make the analogies memorable and easy to visualize.

Examples of good analogies:
- "A solar panel is like a leaf on a tree..."
- "A battery is like a water tank..."
- "An inverter is like a translator..."

Concept to create analogies for: {concept}

Provide just the analogies, one per line:"""

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse analogies from response
            analogies = [line.strip() for line in response_text.split('\n') if line.strip() and not line.startswith('#')]
            return analogies[:3]  # Limit to 3 analogies
            
        except Exception as e:
            print(f"Creative analogies error: {e}")
            return []
    
    async def _get_personalized_learning_path(self, concept: str, user_level: str) -> Dict[str, Any]:
        """Get personalized learning path using OpenRouter"""
        try:
            llm = self.llm_manager.get_llm(self.llm_tasks['personalized_teaching'])
            if not llm:
                return {}
            
            prompt = f"""Create a personalized learning path for someone at {user_level} level who wants to understand "{concept}" in solar energy.

Provide:
1. Prerequisites (what they should know first)
2. Next steps (what to learn after this)
3. Practical exercises or observations they can do
4. Common misconceptions to avoid
5. Real-world applications in Nigeria

Current concept: {concept}
User level: {user_level}

Format as JSON:
{{
  "prerequisites": [],
  "next_steps": [],
  "practical_exercises": [],
  "misconceptions": [],
  "real_world_applications": []
}}"""

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                import json
                return json.loads(response_text)
            except:
                return {
                    'prerequisites': [f'Basic understanding of electricity'],
                    'next_steps': [f'Learn more about solar system components'],
                    'practical_exercises': [f'Observe how {concept} works in real systems'],
                    'misconceptions': [],
                    'real_world_applications': [f'{concept} is used in Nigerian solar installations']
                }
                
        except Exception as e:
            print(f"Personalized learning path error: {e}")
            return {}
    
    async def generate_educational_content_with_llm(self, topic: str, content_type: str = "explanation", user_level: str = "beginner") -> Dict[str, Any]:
        """Generate educational content using appropriate LLM based on content type"""
        try:
            # Choose LLM based on content type
            if content_type == "quick_answer":
                llm_key = self.llm_tasks['quick_explanations']
            elif content_type == "technical_guide":
                llm_key = self.llm_tasks['technical_analysis']
            elif content_type == "creative_explanation":
                llm_key = self.llm_tasks['creative_learning']
            elif content_type == "comprehensive_lesson":
                llm_key = self.llm_tasks['personalized_teaching']
            else:
                llm_key = self.llm_tasks['content_generation']
            
            llm = self.llm_manager.get_llm(llm_key)
            if not llm:
                return {'success': False, 'error': 'LLM not available'}
            
            prompt = f"""Create educational content about "{topic}" in solar energy.

Content Type: {content_type}
User Level: {user_level}
Context: Nigerian solar energy market

Requirements:
- Make it engaging and informative
- Include practical examples
- Use appropriate complexity for {user_level} level
- Include Nigerian context where relevant
- Structure it clearly with headings if needed

Topic: {topic}"""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'success': True,
                'topic': topic,
                'content_type': content_type,
                'user_level': user_level,
                'content': content,
                'llm_used': llm_key,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Educational content generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_content': f"I can help you learn about {topic}. What specific aspect interests you?"
            }
    
    def generate_detailed_explanation(self, topic: str, user_level: str = "beginner", 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate detailed educational explanation with comprehensive breakdown like system sizing"""
        try:
            if not self.llm_manager:
                return self._fallback_detailed_explanation(topic, user_level)
            
            # Generate different sections using different LLMs
            sections = {}
            
            # 1. Quick Overview (Groq Llama3)
            sections['overview'] = self._generate_overview_section(topic, user_level)
            
            # 2. Technical Details (Groq Mixtral) - only for intermediate/advanced
            if user_level in ['intermediate', 'advanced']:
                sections['technical'] = self._generate_technical_section(topic, user_level)
            
            # 3. Practical Examples (HuggingFace)
            sections['examples'] = self._generate_examples_section(topic, user_level, context)
            
            # 4. Cost Analysis (if applicable)
            if any(keyword in topic.lower() for keyword in ['cost', 'price', 'budget', 'system', 'sizing', 'battery', 'panel', 'inverter']):
                sections['cost_analysis'] = self._generate_cost_analysis_section(topic, context)
            
            # 5. Recommendations (OpenRouter Claude)
            sections['recommendations'] = self._generate_recommendations_section(topic, user_level, context)
            
            # Combine into comprehensive explanation
            detailed_explanation = self._format_detailed_explanation(topic, sections, user_level)
            
            return {
                'success': True,
                'topic': topic,
                'user_level': user_level,
                'sections': sections,
                'detailed_explanation': detailed_explanation,
                'learning_time_minutes': self._estimate_learning_time(detailed_explanation),
                'next_topics': self._suggest_next_topics(topic)
            }
            
        except Exception as e:
            print(f"Detailed explanation error: {e}")
            return self._fallback_detailed_explanation(topic, user_level)
    
    def _generate_overview_section(self, topic: str, user_level: str) -> str:
        """Generate overview section using fast LLM"""
        try:
            llm = self.llm_manager.get_llm('groq_llama3')
            if not llm:
                return f"**Overview:** {topic} is an important concept in solar energy systems."
            
            prompt = f"""Hey there! Let's talk about '{topic}' in a way that makes sense for someone at {user_level} level.

I want you to be like a friendly neighbor who knows about solar - warm, encouraging, and easy to understand!

**Explain:**
- What '{topic}' is (in simple, everyday words)
- Why it's important for your solar system (make it relatable!)
- The cool benefits you'll get from understanding this

**Your tone should be:**
- Friendly and conversational (like chatting over coffee)
- Use "you" and "your" to make it personal
- Include encouraging phrases like "Don't worry, it's simpler than it sounds!"
- Keep it under 100 words but make every word count!
- Use bullet points for easy reading"""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return f"**Overview:**\n{content}"
            
        except Exception as e:
            print(f"Overview generation error: {e}")
            return f"**Overview:** {topic} is an important concept in solar energy systems."
    
    def _generate_technical_section(self, topic: str, user_level: str) -> str:
        """Generate technical details using complex reasoning LLM"""
        try:
            llm = self.llm_manager.get_llm('groq_mixtral')
            if not llm:
                return ""
            
            prompt = f"""Provide technical details about '{topic}' for {user_level} level users.

Include:
- Technical specifications or parameters
- How it works (technical process)
- Industry standards or best practices
- Performance metrics or calculations

Format with clear headings and bullet points. Be precise but accessible."""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return f"**Technical Details:**\n{content}"
            
        except Exception as e:
            print(f"Technical section error: {e}")
            return ""
    
    def _generate_examples_section(self, topic: str, user_level: str, context: Dict[str, Any] = None) -> str:
        """Generate practical examples using content generation LLM"""
        try:
            llm = self.llm_manager.get_llm('huggingface')
            if not llm:
                return self._fallback_examples(topic)
            
            context_str = ""
            if context:
                if context.get('appliances'):
                    context_str += f"User has: {', '.join([app.get('name', 'appliance') for app in context['appliances']])}\n"
                if context.get('budget'):
                    context_str += f"Budget: ₦{context['budget']:,}\n"
                if context.get('location'):
                    context_str += f"Location: {context['location'].get('state', 'Nigeria')}\n"
            
            prompt = f"""I want you to be like that helpful friend who's already installed solar and is sharing real stories!

**Topic:** {topic}
{context_str}

**Share 2-3 real-life examples that feel like stories from actual Nigerian families:**

**Make each example:**
- Feel like a real family's experience (use names like "Adebayo family in Lagos" or "Mrs. Okafor in Abuja")
- Include specific numbers that people can relate to
- Show the before/after or the "aha!" moment
- Use everyday language and situations Nigerians face
- Be encouraging and show positive outcomes

**Your tone:**
- Warm and storytelling (like sharing good news with a friend!)
- Use phrases like "Here's what happened when...", "The amazing thing was...", "They were surprised to find..."
- Make it feel real and achievable
- Include little details that make it relatable

Format as numbered stories with engaging titles!"""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return f"**Practical Examples:**\n{content}"
            
        except Exception as e:
            print(f"Examples generation error: {e}")
            return self._fallback_examples(topic)
    
    def _generate_cost_analysis_section(self, topic: str, context: Dict[str, Any] = None) -> str:
        """Generate cost analysis section"""
        try:
            llm = self.llm_manager.get_llm('openrouter_claude')
            if not llm:
                return ""
            
            context_str = ""
            if context:
                if context.get('budget'):
                    context_str += f"User Budget: ₦{context['budget']:,}\n"
                if context.get('system_size'):
                    context_str += f"System Size: {context['system_size']}kW\n"
            
            prompt = f"""Provide cost analysis for '{topic}' in Nigerian solar market.

{context_str}

Include:
- Typical price ranges in Nigerian Naira
- Factors affecting cost
- Cost-saving tips
- Budget recommendations

Be specific with numbers and realistic about Nigerian market prices."""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return f"**Cost Analysis:**\n{content}"
            
        except Exception as e:
            print(f"Cost analysis error: {e}")
            return ""
    
    def _generate_recommendations_section(self, topic: str, user_level: str, context: Dict[str, Any] = None) -> str:
        """Generate recommendations using advanced reasoning LLM"""
        try:
            llm = self.llm_manager.get_llm('openrouter_claude')
            if not llm:
                return self._fallback_recommendations(topic, user_level)
            
            context_str = ""
            if context:
                context_str = f"User Context: {json.dumps(context, indent=2)}\n"
            
            prompt = f"""Generate smart recommendations for '{topic}' for a {user_level} level user.

{context_str}

Provide:
1. **BEST PRACTICES** (3-4 actionable tips)
2. **COMMON MISTAKES TO AVOID** (2-3 warnings)
3. **NEXT STEPS** (what to do next)

Focus on Nigerian solar market context and be practical."""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return f"**Smart Recommendations:**\n{content}"
            
        except Exception as e:
            print(f"Recommendations error: {e}")
            return self._fallback_recommendations(topic, user_level)
    
    def _format_detailed_explanation(self, topic: str, sections: Dict[str, str], user_level: str) -> str:
        """Format all sections into comprehensive explanation"""
        explanation_parts = [
            f"# **Complete Guide: {topic.title()}**",
            f"*Tailored for {user_level.title()} Level*\n"
        ]
        
        # Add sections in logical order
        section_order = ['overview', 'technical', 'examples', 'cost_analysis', 'recommendations']
        
        for section_key in section_order:
            if section_key in sections and sections[section_key]:
                explanation_parts.append(sections[section_key])
                explanation_parts.append("")  # Add spacing
        
        # Add footer
        explanation_parts.append("---")
        explanation_parts.append("**Need more help?** Use the chat interface for specific questions!")
        
        return "\n".join(explanation_parts)
    
    def _estimate_learning_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        word_count = len(content.split())
        return max(2, word_count // 200)  # Assume 200 words per minute
    
    def _suggest_next_topics(self, current_topic: str) -> List[str]:
        """Suggest related topics to learn next"""
        topic_map = {
            'solar panels': ['inverters', 'batteries', 'system sizing'],
            'batteries': ['charge controllers', 'system monitoring', 'maintenance'],
            'inverters': ['solar panels', 'system wiring', 'grid connection'],
            'system sizing': ['cost estimation', 'component selection', 'installation'],
            'cost estimation': ['financing options', 'payback calculation', 'maintenance costs']
        }
        
        return topic_map.get(current_topic.lower(), ['system basics', 'component overview', 'installation guide'])
    
    def _fallback_detailed_explanation(self, topic: str, user_level: str) -> Dict[str, Any]:
        """Fallback when LLMs are not available"""
        return {
            'success': True,
            'topic': topic,
            'user_level': user_level,
            'detailed_explanation': f"""# **Guide: {topic.title()}**

**Overview:**
{topic} is an important component in solar energy systems that helps optimize performance and efficiency.

**Key Points:**
• Essential for solar system operation
• Affects system performance and cost
• Requires proper sizing and selection
• Important for Nigerian solar installations

**Recommendations:**
• Consult with solar professionals
• Consider your specific energy needs
• Factor in local climate conditions
• Plan for future expansion

**Need more help?** Use the chat interface for specific questions!""",
            'learning_time_minutes': 3,
            'next_topics': self._suggest_next_topics(topic)
        }
    
    def _fallback_examples(self, topic: str) -> str:
        """Fallback examples when LLM not available"""
        return f"""**Practical Examples:**
1. **Residential Application**: {topic} in a typical Nigerian home setup
2. **Commercial Use**: How {topic} works in business installations
3. **Cost Consideration**: Budget-friendly options for {topic}"""
    
    def _fallback_recommendations(self, topic: str, user_level: str) -> str:
        """Fallback recommendations when LLM not available"""
        return f"""**Smart Recommendations:**

**BEST PRACTICES:**
• Research thoroughly before making decisions about {topic}
• Consult with certified solar installers
• Consider long-term performance and warranty

**COMMON MISTAKES TO AVOID:**
• Undersizing or oversizing components
• Ignoring local climate factors

**NEXT STEPS:**
• Get professional consultation
• Compare multiple options
• Plan your solar system carefully"""