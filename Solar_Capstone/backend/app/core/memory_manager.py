#!/usr/bin/env python3
"""
Memory Manager for Solar System Recommendation Platform
Handles conversation memory, context tracking, and user preferences
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ConversationMemory:
    """Conversation memory structure"""
    user_id: str
    session_id: str
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    preferences: Dict[str, Any]
    last_updated: datetime
    created_at: datetime

@dataclass
class UserPreferences:
    """User preferences structure"""
    user_id: str
    budget_range: str
    preferred_brands: List[str]
    system_type: str  # 'grid_tie', 'off_grid', 'hybrid'
    location: Dict[str, Any]
    appliances: List[Dict[str, Any]]
    communication_preferences: Dict[str, Any]
    learning_level: str  # 'beginner', 'intermediate', 'advanced'
    last_updated: datetime

class MemoryManager:
    """
    Memory Manager for conversation context and user preferences
    Uses local file storage for simplicity (can be upgraded to database)
    """
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = storage_path
        self.conversations_path = os.path.join(storage_path, "conversations")
        self.preferences_path = os.path.join(storage_path, "preferences")
        
        # Create directories if they don't exist
        os.makedirs(self.conversations_path, exist_ok=True)
        os.makedirs(self.preferences_path, exist_ok=True)
        
        # In-memory cache for active sessions
        self.active_conversations = {}
        self.user_preferences_cache = {}
        
        print(f"MemoryManager initialized with storage at: {storage_path}")
    
    def get_user_id(self, session_data: Dict[str, Any]) -> str:
        """Generate consistent user ID from session data"""
        # Use IP + user agent + timestamp hash for anonymous users
        identifier = f"{session_data.get('ip', 'unknown')}_{session_data.get('user_agent', 'unknown')}"
        return hashlib.md5(identifier.encode()).hexdigest()[:12]
    
    def get_session_id(self, user_id: str) -> str:
        """Generate session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_id}_{timestamp}"
    
    def get_conversation_memory(self, user_id: str, session_id: str) -> Optional[ConversationMemory]:
        """Get conversation memory for a user session"""
        try:
            # Check in-memory cache first
            cache_key = f"{user_id}_{session_id}"
            if cache_key in self.active_conversations:
                return self.active_conversations[cache_key]
            
            # Load from file
            file_path = os.path.join(self.conversations_path, f"{cache_key}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                
                memory = ConversationMemory(**data)
                self.active_conversations[cache_key] = memory
                return memory
            
            return None
            
        except Exception as e:
            print(f"Error loading conversation memory: {e}")
            return None
    
    def save_conversation_memory(self, memory: ConversationMemory) -> bool:
        """Save conversation memory"""
        try:
            cache_key = f"{memory.user_id}_{memory.session_id}"
            
            # Update in-memory cache
            memory.last_updated = datetime.now()
            self.active_conversations[cache_key] = memory
            
            # Save to file
            file_path = os.path.join(self.conversations_path, f"{cache_key}.json")
            data = asdict(memory)
            
            # Convert datetime objects to strings for JSON serialization
            data['last_updated'] = memory.last_updated.isoformat()
            data['created_at'] = memory.created_at.isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving conversation memory: {e}")
            return False
    
    def create_conversation_memory(self, user_id: str, session_id: str) -> ConversationMemory:
        """Create new conversation memory"""
        now = datetime.now()
        memory = ConversationMemory(
            user_id=user_id,
            session_id=session_id,
            messages=[],
            context={},
            preferences={},
            last_updated=now,
            created_at=now
        )
        
        self.save_conversation_memory(memory)
        return memory
    
    def add_message_to_memory(self, user_id: str, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add message to conversation memory"""
        try:
            memory = self.get_conversation_memory(user_id, session_id)
            if not memory:
                memory = self.create_conversation_memory(user_id, session_id)
            
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            memory.messages.append(message)
            
            # Keep only last 50 messages to prevent memory bloat
            if len(memory.messages) > 50:
                memory.messages = memory.messages[-50:]
            
            return self.save_conversation_memory(memory)
            
        except Exception as e:
            print(f"Error adding message to memory: {e}")
            return False
    
    def update_conversation_context(self, user_id: str, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update conversation context"""
        try:
            memory = self.get_conversation_memory(user_id, session_id)
            if not memory:
                memory = self.create_conversation_memory(user_id, session_id)
            
            memory.context.update(context_updates)
            return self.save_conversation_memory(memory)
            
        except Exception as e:
            print(f"Error updating conversation context: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences"""
        try:
            # Check cache first
            if user_id in self.user_preferences_cache:
                return self.user_preferences_cache[user_id]
            
            # Load from file
            file_path = os.path.join(self.preferences_path, f"{user_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert datetime string back to datetime object
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                
                preferences = UserPreferences(**data)
                self.user_preferences_cache[user_id] = preferences
                return preferences
            
            return None
            
        except Exception as e:
            print(f"Error loading user preferences: {e}")
            return None
    
    def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """Save user preferences"""
        try:
            # Update cache
            preferences.last_updated = datetime.now()
            self.user_preferences_cache[preferences.user_id] = preferences
            
            # Save to file
            file_path = os.path.join(self.preferences_path, f"{preferences.user_id}.json")
            data = asdict(preferences)
            
            # Convert datetime object to string for JSON serialization
            data['last_updated'] = preferences.last_updated.isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving user preferences: {e}")
            return False
    
    def create_user_preferences(self, user_id: str, initial_data: Dict[str, Any] = None) -> UserPreferences:
        """Create new user preferences"""
        preferences = UserPreferences(
            user_id=user_id,
            budget_range=initial_data.get('budget_range', 'medium') if initial_data else 'medium',
            preferred_brands=initial_data.get('preferred_brands', []) if initial_data else [],
            system_type=initial_data.get('system_type', 'hybrid') if initial_data else 'hybrid',
            location=initial_data.get('location', {}) if initial_data else {},
            appliances=initial_data.get('appliances', []) if initial_data else [],
            communication_preferences=initial_data.get('communication_preferences', {}) if initial_data else {},
            learning_level=initial_data.get('learning_level', 'beginner') if initial_data else 'beginner',
            last_updated=datetime.now()
        )
        
        self.save_user_preferences(preferences)
        return preferences
    
    def update_user_preferences(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            preferences = self.get_user_preferences(user_id)
            if not preferences:
                preferences = self.create_user_preferences(user_id, updates)
                return True
            
            # Update fields
            for key, value in updates.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
            
            return self.save_user_preferences(preferences)
            
        except Exception as e:
            print(f"Error updating user preferences: {e}")
            return False
    
    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        try:
            memory = self.get_conversation_memory(user_id, session_id)
            if memory and memory.messages:
                return memory.messages[-limit:] if limit else memory.messages
            return []
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up conversations older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            for filename in os.listdir(self.conversations_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.conversations_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        last_updated = datetime.fromisoformat(data['last_updated'])
                        if last_updated < cutoff_date:
                            os.remove(file_path)
                            cleaned_count += 1
                            
                            # Remove from cache if present
                            cache_key = filename.replace('.json', '')
                            if cache_key in self.active_conversations:
                                del self.active_conversations[cache_key]
                                
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
            
            print(f"Cleaned up {cleaned_count} old conversations")
            return cleaned_count
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            conversation_count = len([f for f in os.listdir(self.conversations_path) if f.endswith('.json')])
            preferences_count = len([f for f in os.listdir(self.preferences_path) if f.endswith('.json')])
            
            return {
                'active_conversations': len(self.active_conversations),
                'total_conversations': conversation_count,
                'cached_preferences': len(self.user_preferences_cache),
                'total_preferences': preferences_count,
                'storage_path': self.storage_path
            }
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}

# Global memory manager instance
memory_manager = MemoryManager()
