"""
Communication Agent for Solar Capstone Project
Handles user-installer communication via email and web messaging
"""

from typing import Dict, List, Optional, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pymongo
import os
from pydantic import BaseModel

class CommunicationAgent:
    """
    Multi-LLM Powered Communication Agent for Solar Systems
    
    Uses all 4 LLMs strategically:
    - Groq Llama3: Fast message generation and quick responses
    - Groq Mixtral: Complex communication analysis and context understanding
    - HuggingFace: Knowledge-based content generation and templates
    - Replicate: Creative messaging and engaging content
    - OpenRouter: Advanced communication strategies and personalization
    """
    
    def __init__(self):
        self.agent_name = "CommunicationAgent"
        self.agent_type = "communication"
        self.status = "active"
        self.version = "2.0.0"
        
        # Initialize LLM Manager with all 4 LLMs
        try:
            from ..core.llm_manager import StreamlinedLLMManager
            self.llm_manager = StreamlinedLLMManager()
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
            from llm_manager import StreamlinedLLMManager
            self.llm_manager = StreamlinedLLMManager()
        
        # Multi-LLM task assignment for communication
        self.llm_tasks = {
            'quick_messages': 'groq_llama3',          # Fast message generation
            'complex_analysis': 'groq_mixtral',       # Communication analysis
            'content_templates': 'huggingface',       # Template-based content
            'creative_messaging': 'replicate',        # Engaging content
            'personalized_communication': 'openrouter_claude' # Advanced strategies
        }
        
        print(f"📧 {self.agent_name} v{self.version} initialized with Multi-LLM System:")
        available_llms = self.llm_manager.get_available_providers()
        for llm in available_llms:
            print(f"   ✅ {llm}")
        print(f"   💬 Communication Tasks: {len(self.llm_tasks)}")
        
        # Database connection (use your existing MongoDB)
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017")
            self.db = self.client.solar_recommender
            print(f" {self.agent_name} connected to MongoDB")
        except Exception as e:
            print(f" {self.agent_name} MongoDB connection failed: {e}")
            self.db = None
        
        # Email configuration (use your Gmail)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email = os.getenv("GMAIL_EMAIL", "your_email@gmail.com")
        self.password = os.getenv("GMAIL_APP_PASSWORD", "your_app_password")
        
        print(f" {self.agent_name} initialized successfully")
 
    def handle_quote_request(self, user_data: Dict, system_requirements: Dict) -> Dict:
        """Handle quote request from user to installers"""
        try:
            # 1. Store quote request in database
            quote_request = {
                "user_id": user_data.get("user_id", "demo_user"),
                "user_name": user_data.get("name", "Unknown User"),
                "user_email": user_data.get("email", "user@example.com"),
                "user_phone": user_data.get("phone", "N/A"),
                "location": user_data.get("location", "Unknown Location"),
                "system_requirements": system_requirements,
                "status": "pending",
                "created_at": datetime.now(),
                "responses": []
            }
            
            if self.db:
                quote_id = self.db.quote_requests.insert_one(quote_request).inserted_id
            else:
                quote_id = "demo_quote_id"
            
            # 2. Find installers in the area
            installers = self.find_installers_in_area(user_data.get("location", "Unknown"))
            
            # 3. Send emails to installers
            email_results = []
            for installer in installers:
                email_result = self.send_quote_request_email(installer, quote_request)
                email_results.append(email_result)
            
            # 4. Create initial message thread
            thread_result = self.create_message_thread(str(quote_id), user_data.get("user_id", "demo_user"), "quote_request")
            
            return {
                "success": True,
                "quote_id": str(quote_id),
                "installers_notified": len(installers),
                "email_results": email_results,
                "thread_id": thread_result.get("thread_id"),
                "message": "Quote request sent to installers successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process quote request"
            }
 
    def find_installers_in_area(self, location: str) -> List[Dict]:
        """Find installers in the user's area"""
        try:
            if not self.db:
                # Return demo installers if no database
                return [
                    {
                        "_id": "demo_installer_1",
                        "business_name": "Demo Solar Installer",
                        "email": "installer@demo.com",
                        "service_areas": ["Lagos", "Abuja", "Kano"]
                    }
                ]
            
            # Query installers by location
            installers = self.db.installers.find({
                "service_areas": {"$regex": location, "$options": "i"},
                "status": "active"
            })
            
            return list(installers)
        except Exception as e:
            print(f"Error finding installers: {e}")
            return []
 
    def send_quote_request_email(self, installer: Dict, quote_request: Dict) -> Dict:
        """Send quote request email to installer"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = installer["email"]
            msg['Subject'] = f"New Solar Quote Request - {quote_request['user_name']}"
            
            body = f"""
            Hello {installer['business_name']},
            
            A new solar quote request has been submitted in your service area:
            
            Customer Details:
            - Name: {quote_request['user_name']}
            - Email: {quote_request['user_email']}
            - Phone: {quote_request['user_phone']}
            - Location: {quote_request['location']}
            
            System Requirements:
            - Power: {quote_request['system_requirements'].get('power', 'N/A')}W
            - Budget: {quote_request['system_requirements'].get('budget', 'N/A')}
            - Timeline: {quote_request['system_requirements'].get('timeline', 'N/A')}
            
            To respond to this quote request:
            1. Reply to this email directly
            2. Or visit your installer dashboard
            3. Or use our web messaging system
            
            Best regards,
            Solar Capstone Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            
            return {
                "success": True,
                "installer_id": installer["_id"],
                "email_sent": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "installer_id": installer["_id"],
                "error": str(e)
            }
 
    def create_message_thread(self, quote_id: str, user_id: str, thread_type: str) -> Dict:
        """Create a new message thread"""
        try:
            thread = {
                "quote_id": quote_id,
                "user_id": user_id,
                "thread_type": thread_type,
                "status": "active",
                "created_at": datetime.now(),
                "messages": []
            }
            
            if self.db:
                thread_id = self.db.message_threads.insert_one(thread).inserted_id
            else:
                thread_id = "demo_thread_id"
            
            return {
                "success": True,
                "thread_id": str(thread_id)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
 
    def send_message(self, sender_id: str, recipient_id: str, message: str, thread_id: str = None) -> Dict:
        """Send a message between user and installer"""
        try:
            message_data = {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message": message,
                "thread_id": thread_id,
                "timestamp": datetime.now(),
                "read": False,
                "message_type": "text"
            }
            
            # Store message in database
            if self.db:
                message_id = self.db.messages.insert_one(message_data).inserted_id
            else:
                message_id = "demo_message_id"
            
            # Send email notification to recipient
            self.send_message_notification(recipient_id, message)
            
            return {
                "success": True,
                "message_id": str(message_id),
                "timestamp": message_data["timestamp"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
 
    def send_message_notification(self, recipient_id: str, message: str) -> Dict:
        """Send email notification for new message"""
        try:
            # Get recipient details
            if self.db:
                recipient = self.db.users.find_one({"_id": recipient_id}) or \
                self.db.installers.find_one({"_id": recipient_id})
            else:
                recipient = {"email": "demo@example.com", "name": "Demo User"}
            
            if not recipient:
                return {"success": False, "error": "Recipient not found"}
            
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = recipient["email"]
            msg['Subject'] = "New Message - Solar Capstone"
            
            body = f"""
            Hello {recipient.get('name', 'User')},
            
            You have received a new message:
            
            "{message[:100]}{'...' if len(message) > 100 else ''}"
            
            To view and respond to this message:
            1. Visit your dashboard
            2. Or reply to this email directly
            
            Best regards,
            Solar Capstone Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
 
    def get_messages(self, user_id: str, thread_id: str = None) -> List[Dict]:
        """Get messages for a user"""
        try:
            if not self.db:
                # Return demo messages
                return [
                    {
                        "sender_id": "demo_installer",
                        "recipient_id": user_id,
                        "message": "Thank you for your interest in our solar services!",
                        "timestamp": datetime.now(),
                        "read": False
                    }
                ]
            
            query = {"$or": [{"sender_id": user_id}, {"recipient_id": user_id}]}
            if thread_id:
                query["thread_id"] = thread_id
            
            messages = self.db.messages.find(query).sort("timestamp", -1)
            
            return list(messages)
            
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []
 
    def mark_message_read(self, message_id: str) -> Dict:
        """Mark a message as read"""
        try:
            if not self.db:
                return {"success": True, "modified": 1}
            
            result = self.db.messages.update_one(
                {"_id": message_id},
                {"$set": {"read": True}}
            )
            
            return {"success": True, "modified": result.modified_count}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
 
    def get_agent_status(self) -> Dict:
        """Get communication agent status"""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status,
            "database_connected": self.db is not None,
            "email_configured": bool(self.email and self.password)
        }