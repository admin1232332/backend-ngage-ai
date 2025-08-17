#!/usr/bin/env python3
"""
Database module for nGAGE AI Feedback Writer
Handles PostgreSQL connections and logging operations
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Numeric
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            # Use local SQLite database if DATABASE_URL is not set
            self.database_url = "sqlite:///ngage_local.db"
        self.engine = None
        self.metadata = MetaData()
        self._initialize_connection()
        self._create_tables()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        if not self.database_url:
            print("⚠️ DATABASE_URL not found. Database logging disabled.")
            return
        
        try:
            # Handle Railway's postgres:// URL format
            if self.database_url.startswith('postgres://'):
                self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
            
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("✅ Database connection established successfully")
            
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            self.engine = None
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                # Create feedback_logs table
                if 'sqlite' in self.database_url.lower():
                    # SQLite syntax
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS feedback_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_context TEXT NOT NULL,
                            selected_tone TEXT NOT NULL CHECK (selected_tone IN ('positive', 'constructive')),
                            selected_style TEXT NOT NULL CHECK (selected_style IN ('balanced', 'formal', 'casual', 'appreciative')),
                            generated_feedback TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            user_ip TEXT,
                            validation_score REAL,
                            context_sentiment TEXT,
                            session_id TEXT
                        )
                    """))
                else:
                    # PostgreSQL syntax
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS feedback_logs (
                            id SERIAL PRIMARY KEY,
                            user_context TEXT NOT NULL,
                            selected_tone VARCHAR(20) NOT NULL CHECK (selected_tone IN ('positive', 'constructive')),
                            selected_style VARCHAR(20) NOT NULL CHECK (selected_style IN ('balanced', 'formal', 'casual', 'appreciative')),
                            generated_feedback TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            user_ip VARCHAR(45),
                            validation_score DECIMAL(3,2),
                            context_sentiment VARCHAR(10),
                            session_id VARCHAR(100)
                        )
                    """))
                
                # Create indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_logs_timestamp ON feedback_logs(timestamp)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_logs_tone ON feedback_logs(selected_tone)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_logs_style ON feedback_logs(selected_style)
                """))
                
                conn.commit()
                print("✅ Database tables created/verified successfully")
                
        except Exception as e:
            print(f"❌ Failed to create database tables: {e}")
    
    def log_feedback(self, 
                    context: str, 
                    tone: str, 
                    style: str, 
                    feedback: str, 
                    user_ip: Optional[str] = None,
                    validation_score: Optional[float] = None,
                    context_sentiment: Optional[str] = None,
                    session_id: Optional[str] = None) -> bool:
        """
        Log feedback generation to database
        
        Args:
            context: User's input context
            tone: Selected feedback tone
            style: Selected feedback style
            feedback: Generated feedback text
            user_ip: User's IP address (optional)
            validation_score: AI validation score (optional)
            context_sentiment: Detected context sentiment (optional)
            session_id: User session ID (optional)
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if not self.engine:
            return False
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO feedback_logs 
                    (user_context, selected_tone, selected_style, generated_feedback, 
                     user_ip, validation_score, context_sentiment, session_id)
                    VALUES (:context, :tone, :style, :feedback, :user_ip, 
                            :validation_score, :context_sentiment, :session_id)
                """), {
                    'context': context[:1000],  # Limit context length
                    'tone': tone,
                    'style': style,
                    'feedback': feedback[:2000],  # Limit feedback length
                    'user_ip': user_ip,
                    'validation_score': validation_score,
                    'context_sentiment': context_sentiment,
                    'session_id': session_id
                })
                conn.commit()
                
            return True
            
        except Exception as e:
            print(f"❌ Database logging failed: {e}")
            return False
    
    def get_analytics(self) -> Optional[Dict]:
        """
        Get basic analytics from feedback logs
        
        Returns:
            Dict with analytics data or None if failed
        """
        if not self.engine:
            return None
        
        try:
            with self.engine.connect() as conn:
                # Total feedback generated
                total_result = conn.execute(text("SELECT COUNT(*) FROM feedback_logs"))
                total = total_result.scalar()
                
                # Most popular tone
                tone_result = conn.execute(text("""
                    SELECT selected_tone, COUNT(*) as count 
                    FROM feedback_logs 
                    GROUP BY selected_tone 
                    ORDER BY count DESC 
                    LIMIT 1
                """))
                popular_tone_row = tone_result.fetchone()
                popular_tone = popular_tone_row[0] if popular_tone_row else None
                
                # Most popular style
                style_result = conn.execute(text("""
                    SELECT selected_style, COUNT(*) as count 
                    FROM feedback_logs 
                    GROUP BY selected_style 
                    ORDER BY count DESC 
                    LIMIT 1
                """))
                popular_style_row = style_result.fetchone()
                popular_style = popular_style_row[0] if popular_style_row else None
                
                # Average validation score
                score_result = conn.execute(text("""
                    SELECT AVG(validation_score) 
                    FROM feedback_logs 
                    WHERE validation_score IS NOT NULL
                """))
                avg_score = score_result.scalar()
                
                # Recent activity (last 7 days)
                if 'sqlite' in self.database_url.lower():
                    recent_result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM feedback_logs 
                        WHERE timestamp >= datetime('now', '-7 days')
                    """))
                else:
                    recent_result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM feedback_logs 
                        WHERE timestamp >= NOW() - INTERVAL '7 days'
                    """))
                recent_count = recent_result.scalar()
                
                # Unique users (based on IP)
                users_result = conn.execute(text("""
                    SELECT COUNT(DISTINCT user_ip) 
                    FROM feedback_logs 
                    WHERE user_ip IS NOT NULL
                """))
                unique_users = users_result.scalar()
                
                return {
                    'total_feedback_generated': total,
                    'most_popular_tone': popular_tone,
                    'most_popular_style': popular_style,
                    'average_quality_score': float(avg_score) if avg_score else None,
                    'recent_activity_7_days': recent_count,
                    'unique_users': unique_users,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Analytics query failed: {e}")
            return None
    
    def get_recent_feedback(self, limit: int = 10) -> Optional[List[Dict]]:
        """
        Get recent feedback entries (without sensitive data)
        
        Args:
            limit: Number of entries to return
            
        Returns:
            List of recent feedback entries or None if failed
        """
        if not self.engine:
            return None
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT selected_tone, selected_style, validation_score, 
                           context_sentiment, timestamp
                    FROM feedback_logs 
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """), {'limit': limit})
                
                entries = []
                for row in result:
                    # Handle timestamp formatting for both SQLite and PostgreSQL
                    timestamp_val = row[4]
                    if timestamp_val:
                        if hasattr(timestamp_val, 'isoformat'):
                            timestamp_str = timestamp_val.isoformat()
                        else:
                            timestamp_str = str(timestamp_val)
                    else:
                        timestamp_str = None
                    
                    entries.append({
                        'tone': row[0],
                        'style': row[1],
                        'validation_score': float(row[2]) if row[2] else None,
                        'context_sentiment': row[3],
                        'timestamp': timestamp_str
                    })
                
                return entries
                
        except Exception as e:
            print(f"❌ Recent feedback query failed: {e}")
            return None
    
    def health_check(self) -> Dict:
        """
        Check database health
        
        Returns:
            Dict with health status
        """
        if not self.engine:
            return {
                'status': 'disconnected',
                'message': 'Database not configured'
            }
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
                # Get table count
                table_result = conn.execute(text("""
                    SELECT COUNT(*) FROM feedback_logs
                """))
                record_count = table_result.scalar()
                
                return {
                    'status': 'connected',
                    'message': 'Database is healthy',
                    'total_records': record_count
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database error: {str(e)}'
            }

# Global database manager instance
db_manager = DatabaseManager()