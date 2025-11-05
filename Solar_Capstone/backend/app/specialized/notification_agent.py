"""
Notification Agent - User Communication and Alerts
Handles user notifications, alerts, and communications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import smtplib
from datetime import datetime, timedelta
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import core infrastructure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from agent_base import BaseAgent

@dataclass
class NotificationTemplate:
    """Notification template"""
    template_id: str
    name: str
    subject: str
    body: str
    notification_type: str # 'email', 'sms', 'push'
    is_active: bool

@dataclass
class Notification:
    """Notification instance"""
    notification_id: str
    user_id: str
    template_id: str
    subject: str
    body: str
    notification_type: str
    status: str # 'pending', 'sent', 'failed'
    created_at: datetime
    sent_at: Optional[datetime]
    retry_count: int

@dataclass
class UserPreference:
    """User notification preferences"""
    user_id: str
    email_enabled: bool
    sms_enabled: bool
    push_enabled: bool
    price_alerts: bool
    system_updates: bool
    educational_content: bool
    frequency: str # 'immediate', 'daily', 'weekly'

class NotificationAgent(BaseAgent):
    """Notification agent for user communications"""
    
    def __init__(self, llm_manager=None, tool_manager=None, nlp_processor=None):
        super().__init__("NotificationAgent", llm_manager, tool_manager, nlp_processor)
        self.notification_templates = []
        self.pending_notifications = []
        self.user_preferences = {}
        self._load_notification_templates()
        self._load_user_preferences()
        self._start_notification_processor()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process notification request"""
        try:
            self.status = 'processing'
            
            # Validate input
            if not self.validate_input(input_data, ['action']):
                return self.create_response(None, False, "Missing action parameter")
            
            action = input_data['action']
            
            if action == 'send_notification':
                return self._send_notification(input_data)
            elif action == 'schedule_notification':
                return self._schedule_notification(input_data)
            elif action == 'get_notification_status':
                return self._get_notification_status(input_data)
            elif action == 'update_user_preferences':
                return self._update_user_preferences(input_data)
            elif action == 'send_bulk_notification':
                return self._send_bulk_notification(input_data)
            else:
                return self.create_response(None, False, f"Unknown action: {action}")
            
        except Exception as e:
            return self.handle_error(e, "Notification processing")
        finally:
            self.status = 'idle'
    
    def _send_notification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single notification"""
        try:
            user_id = input_data.get('user_id')
            template_id = input_data.get('template_id')
            custom_data = input_data.get('custom_data', {})
            notification_type = input_data.get('notification_type', 'email')
            
            if not user_id or not template_id:
                return self.create_response(None, False, "User ID and template ID required")
            
            # Get template
            template = next((t for t in self.notification_templates if t.template_id == template_id), None)
            if not template:
                return self.create_response(None, False, f"Template {template_id} not found")
            
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, UserPreference(
                user_id=user_id,
                email_enabled=True,
                sms_enabled=False,
                push_enabled=False,
                price_alerts=True,
                system_updates=True,
                educational_content=True,
                frequency='immediate'
            ))
            
            # Check if user wants this type of notification
            if notification_type == 'email' and not user_prefs.email_enabled:
                return self.create_response(None, False, "User has disabled email notifications")
            elif notification_type == 'sms' and not user_prefs.sms_enabled:
                return self.create_response(None, False, "User has disabled SMS notifications")
            
            # Create notification
            notification = Notification(
                notification_id=f"notif_{int(time.time())}",
                user_id=user_id,
                template_id=template_id,
                subject=self._format_template(template.subject, custom_data),
                body=self._format_template(template.body, custom_data),
                notification_type=notification_type,
                status='pending',
                created_at=datetime.now(),
                sent_at=None,
                retry_count=0
            )
            
            # Send notification
            success = self._send_notification_immediate(notification)
            
            if success:
                notification.status = 'sent'
                notification.sent_at = datetime.now()
                return self.create_response({
                    'notification_id': notification.notification_id,
                    'status': 'sent',
                    'sent_at': notification.sent_at.isoformat()
                }, True, "Notification sent successfully")
            else:
                notification.status = 'failed'
                return self.create_response(None, False, "Failed to send notification")
            
        except Exception as e:
            return self.handle_error(e, "Sending notification")
    
    def _schedule_notification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a notification for later"""
        try:
            user_id = input_data.get('user_id')
            template_id = input_data.get('template_id')
            schedule_time = input_data.get('schedule_time')
            custom_data = input_data.get('custom_data', {})
            notification_type = input_data.get('notification_type', 'email')
            
            if not all([user_id, template_id, schedule_time]):
                return self.create_response(None, False, "User ID, template ID, and schedule time required")
            
            # Parse schedule time
            try:
                schedule_datetime = datetime.fromisoformat(schedule_time)
            except:
                return self.create_response(None, False, "Invalid schedule time format")
            
            # Create scheduled notification
            notification = Notification(
                notification_id=f"sched_{int(time.time())}",
                user_id=user_id,
                template_id=template_id,
                subject="Scheduled notification",
                body="Scheduled notification body",
                notification_type=notification_type,
                status='scheduled',
                created_at=datetime.now(),
                sent_at=None,
                retry_count=0
            )
            
            # Add to pending notifications
            self.pending_notifications.append(notification)
            
            return self.create_response({
                'notification_id': notification.notification_id,
                'status': 'scheduled',
                'schedule_time': schedule_time
            }, True, "Notification scheduled successfully")
            
        except Exception as e:
            return self.handle_error(e, "Scheduling notification")
    
    def _get_notification_status(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get notification status and statistics"""
        try:
            user_id = input_data.get('user_id')
            
            if user_id:
                # Get notifications for specific user
                user_notifications = [n for n in self.pending_notifications if n.user_id == user_id]
            else:
                # Get all notifications
                user_notifications = self.pending_notifications
            
            # Calculate statistics
            total_notifications = len(user_notifications)
            sent_notifications = len([n for n in user_notifications if n.status == 'sent'])
            failed_notifications = len([n for n in user_notifications if n.status == 'failed'])
            pending_notifications = len([n for n in user_notifications if n.status == 'pending'])
            
            return self.create_response({
                'total_notifications': total_notifications,
                'sent_notifications': sent_notifications,
                'failed_notifications': failed_notifications,
                'pending_notifications': pending_notifications,
                'success_rate': (sent_notifications / total_notifications * 100) if total_notifications > 0 else 0,
                'notifications': [
                    {
                        'notification_id': n.notification_id,
                        'user_id': n.user_id,
                        'status': n.status,
                        'created_at': n.created_at.isoformat(),
                        'sent_at': n.sent_at.isoformat() if n.sent_at else None
                    }
                    for n in user_notifications
                ]
            }, True, "Notification status retrieved")
            
        except Exception as e:
            return self.handle_error(e, "Getting notification status")
    
    def _update_user_preferences(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user notification preferences"""
        try:
            user_id = input_data.get('user_id')
            preferences = input_data.get('preferences', {})
            
            if not user_id:
                return self.create_response(None, False, "User ID required")
            
            # Update or create user preferences
            if user_id in self.user_preferences:
                user_prefs = self.user_preferences[user_id]
                for key, value in preferences.items():
                    if hasattr(user_prefs, key):
                        setattr(user_prefs, key, value)
            else:
                user_prefs = UserPreference(
                    user_id=user_id,
                    email_enabled=preferences.get('email_enabled', True),
                    sms_enabled=preferences.get('sms_enabled', False),
                    push_enabled=preferences.get('push_enabled', False),
                    price_alerts=preferences.get('price_alerts', True),
                    system_updates=preferences.get('system_updates', True),
                    educational_content=preferences.get('educational_content', True),
                    frequency=preferences.get('frequency', 'immediate')
                )
                self.user_preferences[user_id] = user_prefs
            
            return self.create_response({
                'user_id': user_id,
                'preferences': preferences,
                'updated_at': datetime.now().isoformat()
            }, True, "User preferences updated successfully")
            
        except Exception as e:
            return self.handle_error(e, "Updating user preferences")
    
    def _send_bulk_notification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send bulk notifications to multiple users"""
        try:
            user_ids = input_data.get('user_ids', [])
            template_id = input_data.get('template_id')
            custom_data = input_data.get('custom_data', {})
            notification_type = input_data.get('notification_type', 'email')
            
            if not user_ids or not template_id:
                return self.create_response(None, False, "User IDs and template ID required")
            
            # Get template
            template = next((t for t in self.notification_templates if t.template_id == template_id), None)
            if not template:
                return self.create_response(None, False, f"Template {template_id} not found")
            
            results = []
            success_count = 0
            
            for user_id in user_ids:
                try:
                    # Check user preferences
                    user_prefs = self.user_preferences.get(user_id)
                    if user_prefs and not self._should_send_notification(user_prefs, notification_type):
                        results.append({
                            'user_id': user_id,
                            'status': 'skipped',
                            'reason': 'User preferences disabled'
                        })
                        continue
                    
                    # Create and send notification
                    notification = Notification(
                        notification_id=f"bulk_{int(time.time())}_{user_id}",
                        user_id=user_id,
                        template_id=template_id,
                        subject=self._format_template(template.subject, custom_data),
                        body=self._format_template(template.body, custom_data),
                        notification_type=notification_type,
                        status='pending',
                        created_at=datetime.now(),
                        sent_at=None,
                        retry_count=0
                    )
                    
                    success = self._send_notification_immediate(notification)
                    
                    if success:
                        notification.status = 'sent'
                        notification.sent_at = datetime.now()
                        success_count += 1
                        results.append({
                            'user_id': user_id,
                            'status': 'sent',
                            'sent_at': notification.sent_at.isoformat()
                        })
                    else:
                        notification.status = 'failed'
                        results.append({
                            'user_id': user_id,
                            'status': 'failed',
                            'reason': 'Send failed'
                        })
                    
                except Exception as e:
                    results.append({
                        'user_id': user_id,
                        'status': 'error',
                        'reason': str(e)
                    })
            
            return self.create_response({
                'total_users': len(user_ids),
                'success_count': success_count,
                'success_rate': (success_count / len(user_ids)) * 100,
                'results': results
            }, True, f"Bulk notification sent to {success_count}/{len(user_ids)} users")
            
        except Exception as e:
            return self.handle_error(e, "Sending bulk notification")
    
    def _send_notification_immediate(self, notification: Notification) -> bool:
        """Send notification immediately"""
        try:
            if notification.notification_type == 'email':
                return self._send_email_notification(notification)
            elif notification.notification_type == 'sms':
                return self._send_sms_notification(notification)
            elif notification.notification_type == 'push':
                return self._send_push_notification(notification)
            else:
                return False
            
        except Exception as e:
            self.log_activity(f"Error sending notification: {e}", 'warning')
            return False
    
    def _send_email_notification(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Mock email sending - in practice would use actual SMTP
            self.log_activity(f"Email sent to user {notification.user_id}: {notification.subject}")
            return True
            
        except Exception as e:
            self.log_activity(f"Error sending email: {e}", 'warning')
            return False
    
    def _send_sms_notification(self, notification: Notification) -> bool:
        """Send SMS notification"""
        try:
            # Mock SMS sending - in practice would use SMS service
            self.log_activity(f"SMS sent to user {notification.user_id}: {notification.subject}")
            return True
            
        except Exception as e:
            self.log_activity(f"Error sending SMS: {e}", 'warning')
            return False
    
    def _send_push_notification(self, notification: Notification) -> bool:
        """Send push notification"""
        try:
            # Mock push notification - in practice would use push service
            self.log_activity(f"Push notification sent to user {notification.user_id}: {notification.subject}")
            return True
            
        except Exception as e:
            self.log_activity(f"Error sending push notification: {e}", 'warning')
            return False
    
    def _format_template(self, template: str, custom_data: Dict[str, Any]) -> str:
        """Format notification template with custom data"""
        try:
            formatted = template
            
            for key, value in custom_data.items():
                placeholder = f"{{{key}}}"
                formatted = formatted.replace(placeholder, str(value))
            
            return formatted
            
        except Exception as e:
            self.log_activity(f"Error formatting template: {e}", 'warning')
            return template
    
    def _should_send_notification(self, user_prefs: UserPreference, notification_type: str) -> bool:
        """Check if notification should be sent based on user preferences"""
        if notification_type == 'email':
            return user_prefs.email_enabled
        elif notification_type == 'sms':
            return user_prefs.sms_enabled
        elif notification_type == 'push':
            return user_prefs.push_enabled
        else:
            return True
    
    def _start_notification_processor(self):
        """Start background notification processor"""
        try:
            # Start processor thread
            processor_thread = threading.Thread(target=self._notification_processor_loop, daemon=True)
            processor_thread.start()
            
            self.log_activity("Notification processor started")
            
        except Exception as e:
            self.log_activity(f"Error starting notification processor: {e}", 'warning')
    
    def _notification_processor_loop(self):
        """Background notification processor loop"""
        while True:
            try:
                self._process_pending_notifications()
                time.sleep(30) # Check every 30 seconds
            except Exception as e:
                self.log_activity(f"Error in notification processor loop: {e}", 'warning')
                time.sleep(30)
    
    def _process_pending_notifications(self):
        """Process pending notifications"""
        try:
            current_time = datetime.now()
            
            for notification in self.pending_notifications[:]: # Copy list to avoid modification during iteration
                if notification.status == 'scheduled':
                    # Check if it's time to send
                    if notification.created_at <= current_time:
                        success = self._send_notification_immediate(notification)
                        
                        if success:
                            notification.status = 'sent'
                            notification.sent_at = current_time
                        else:
                            notification.status = 'failed'
                
                elif notification.status == 'failed' and notification.retry_count < 3:
                    # Retry failed notifications
                    notification.retry_count += 1
                    success = self._send_notification_immediate(notification)
                    
                    if success:
                        notification.status = 'sent'
                        notification.sent_at = current_time
                    else:
                        notification.status = 'failed'
                
                # Remove old notifications (older than 7 days)
                cutoff_date = current_time - timedelta(days=7)
                self.pending_notifications = [
                    n for n in self.pending_notifications
                    if n.created_at >= cutoff_date
                ]
            
        except Exception as e:
            self.log_activity(f"Error processing pending notifications: {e}", 'warning')
    
    def _load_notification_templates(self):
        """Load notification templates"""
        try:
            # Default notification templates
            self.notification_templates = [
                NotificationTemplate(
                    template_id='price_alert',
                    name='Price Alert',
                    subject='Price Alert: {product_name}',
                    body='The price of {product_name} has dropped to ₦{current_price:,.0f}. Your target price was ₦{target_price:,.0f}.',
                    notification_type='email',
                    is_active=True
                ),
                NotificationTemplate(
                    template_id='system_update',
                    name='System Update',
                    subject='Solar System Update',
                    body='Your solar system analysis is complete. Check your dashboard for recommendations.',
                    notification_type='email',
                    is_active=True
                ),
                NotificationTemplate(
                    template_id='educational_content',
                    name='Educational Content',
                    subject='New Solar Learning Content',
                    body='New educational content is available: {content_title}. Learn more about {topic}.',
                    notification_type='email',
                    is_active=True
                ),
                NotificationTemplate(
                    template_id='welcome',
                    name='Welcome',
                    subject='Welcome to Solar System Analyzer',
                    body='Welcome! Get started with your solar system analysis.',
                    notification_type='email',
                    is_active=True
                )
            ]
            
            self.log_activity("Notification templates loaded successfully")
            
        except Exception as e:
            self.log_activity(f"Error loading notification templates: {e}", 'warning')
    
    def _load_user_preferences(self):
        """Load user preferences"""
        try:
            # Initialize with default preferences
            self.user_preferences = {}
            
            self.log_activity("User preferences loaded successfully")
            
        except Exception as e:
            self.log_activity(f"Error loading user preferences: {e}", 'warning')