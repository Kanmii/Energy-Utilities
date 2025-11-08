"""
NLP Processor for Solar Recommender System
Handles natural language understanding, intent classification, and entity extraction
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


class NLPProcessor:
    """Natural Language Processing for solar system interactions

    This implementation intentionally avoids a dependency on spaCy by
    falling back to lightweight NLTK-based processing. It's suitable for
    basic intent/keyword extraction and sentiment analysis while the
    full spaCy model is unavailable.
    """

    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.solar_keywords = self._load_solar_keywords()
        self.intent_patterns = self._load_intent_patterns()
        self._initialize_nlp()

    def _initialize_nlp(self):
        """Initialize NLP tools (NLTK-based)."""
        # Keep self.nlp = None to indicate spaCy is not used
        self.nlp = None

        try:
            # Ensure the VADER lexicon is available for sentiment analysis
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            print("SUCCESS: Sentiment analyzer initialized")
        except Exception as e:
            print(f"WARNING: Sentiment analyzer failed: {e}")
            self.sentiment_analyzer = None
    
    def _load_solar_keywords(self) -> Dict[str, List[str]]:
        """Load solar energy related keywords"""
        return {
            'solar_terms': [
                'solar', 'photovoltaic', 'pv', 'panel', 'panels', 'battery', 'batteries',
                'inverter', 'charge controller', 'grid-tie', 'off-grid', 'hybrid',
                'energy', 'electricity', 'power', 'watt', 'kilowatt', 'kwh'
            ],
            'appliance_terms': [
                'appliance', 'appliances', 'refrigerator', 'fridge', 'television', 'tv',
                'air conditioner', 'ac', 'fan', 'light', 'lights', 'computer', 'laptop',
                'phone', 'charger', 'washing machine', 'dryer', 'microwave', 'oven'
            ],
            'location_terms': [
                'location', 'address', 'city', 'state', 'region', 'area', 'place',
                'lagos', 'abuja', 'kano', 'ibadan', 'port harcourt', 'benin'
            ],
            'financial_terms': [
                'budget', 'cost', 'price', 'expensive', 'cheap', 'affordable',
                'naira', 'dollar', 'money', 'pay', 'payment', 'installment'
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent classification patterns"""
        return {
            'system_design': [
                r'design.*system', r'calculate.*system', r'size.*system',
                r'how much.*power', r'what.*system', r'need.*solar'
            ],
            'appliance_analysis': [
                r'add.*appliance', r'remove.*appliance', r'what.*appliance',
                r'how much.*energy', r'power.*consumption', r'energy.*usage'
            ],
            'location_analysis': [
                r'location.*solar', r'solar.*potential', r'sun.*hours',
                r'weather.*solar', r'climate.*solar', r'irradiance'
            ],
            'cost_analysis': [
                r'cost.*solar', r'price.*system', r'budget.*solar',
                r'how much.*cost', r'expensive.*solar', r'payback'
            ],
            'component_recommendation': [
                r'recommend.*component', r'best.*panel', r'best.*battery',
                r'which.*inverter', r'component.*recommendation', r'brand.*recommendation'
            ],
            'education': [
                r'explain.*solar', r'what.*solar', r'how.*solar.*work',
                r'teach.*solar', r'learn.*solar', r'understand.*solar'
            ],
            'general_question': [
                r'help', r'question', r'ask', r'information', r'advice'
            ]
        }
    
    def process_input(self, text: str) -> Dict[str, Any]:
        """Process user input and extract information"""
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Extract entities
            entities = self._extract_entities(cleaned_text)
            
            # Classify intent
            intent = self._classify_intent(cleaned_text)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(cleaned_text)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(intent, entities, keywords)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'intent': intent,
                'entities': entities,
                'sentiment': sentiment,
                'keywords': keywords,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f" ERROR: NLP processing failed: {e}")
            return {
                'original_text': text,
                'cleaned_text': text,
                'intent': 'general_question',
                'entities': {},
                'sentiment': 'neutral',
                'keywords': [],
                'confidence': 0.5
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            'appliances': [],
            'locations': [],
            'numbers': [],
            'solar_terms': []
        }

        # Extract appliances and solar/location terms by simple substring match
        for category, terms in self.solar_keywords.items():
            if category == 'appliance_terms':
                for term in terms:
                    if term in text:
                        entities['appliances'].append(term)
            elif category == 'location_terms':
                for term in terms:
                    if term in text:
                        entities['locations'].append(term)
            elif category == 'solar_terms':
                for term in terms:
                    if term in text:
                        entities['solar_terms'].append(term)

        # Extract numbers (floats or ints)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        entities['numbers'] = [float(n) for n in numbers]

        return entities
    
    def _classify_intent(self, text: str) -> str:
        """Classify user intent"""
        best_intent = 'general_question'
        best_score = 0
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer:
            return 'neutral'
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            if scores['compound'] >= 0.05:
                return 'positive'
            elif scores['compound'] <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        
        # Tokenize text
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        # Filter for relevant keywords
        for token in tokens:
            if len(token) > 2 and token.isalpha():
                keywords.append(token)
        
        return keywords
    
    def _calculate_confidence(self, intent: str, entities: Dict, keywords: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on entities found
        total_entities = sum(len(entities[key]) for key in entities.keys())
        if total_entities > 0:
            confidence += min(0.3, total_entities * 0.1)
        
        # Boost confidence based on keywords
        if len(keywords) > 0:
            confidence += min(0.2, len(keywords) * 0.05)
        
        # Boost confidence for specific intents
        if intent != 'general_question':
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def extract_requirements(self, text: str) -> Dict[str, Any]:
        """Extract system requirements from text"""
        requirements = {
            'appliances': [],
            'location': None,
            'budget': None,
            'autonomy_days': None,
            'preferences': {}
        }
        
        # Extract appliances
        entities = self._extract_entities(text)
        requirements['appliances'] = entities.get('appliances', [])
        
        # Extract location
        locations = entities.get('locations', [])
        if locations:
            requirements['location'] = locations[0]
        
        # Extract budget information
        numbers = entities.get('numbers', [])
        if numbers:
            # Assume the largest number is budget
            requirements['budget'] = max(numbers)
        
        # Extract autonomy days
        if 'day' in text or 'days' in text:
            for num in numbers:
                if 1 <= num <= 7:  # Reasonable range for autonomy days
                    requirements['autonomy_days'] = int(num)
                    break
        
        return requirements
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get NLP processor status"""
        return {
            'nlp_available': self.nlp is not None,
            'sentiment_available': self.sentiment_analyzer is not None,
            'keywords_loaded': len(self.solar_keywords),
            'intent_patterns_loaded': len(self.intent_patterns),
            'status': 'active'
        }

# Create global instance
nlp_processor = NLPProcessor()