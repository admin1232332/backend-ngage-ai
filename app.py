#!/usr/bin/env python3
"""
Simple Flask server for nGAGE AI Feedback Writer
Connects the HTML frontend to AWS Bedrock
Enhanced with PostgreSQL database logging
"""

import os
import json
import boto3
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from database import db_manager

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class AIFeedbackGenerator:
    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.model_id = os.getenv("AWS_BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Bedrock client"""
        try:
            if not all([self.aws_access_key, self.aws_secret_key]):
                raise ValueError("Missing AWS credentials")
            
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            print("✅ Bedrock client initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Bedrock client: {e}")
            self.client = None
    
    def generate_feedback(self, context, tone, style="balanced"):
        """Generate feedback using AI"""
        if not self.client:
            raise Exception("Bedrock client not initialized")
        
        # Style-specific instructions
        style_instructions = {
            "balanced": "professional and neutral tone",
            "formal": "formal, corporate language with structured sentences",
            "casual": "friendly, conversational tone that's approachable",
            "appreciative": "warm, grateful tone that shows genuine appreciation"
        }
        
        style_instruction = style_instructions.get(style, style_instructions["balanced"])
        
        # Enhanced prompt that handles context-tone mismatches intelligently
        if tone.lower() == "constructive":
            tone_guidance = """
IMPORTANT: Even if the context mentions positive outcomes, focus on providing constructive feedback by:
- Identifying areas for growth or improvement
- Suggesting ways to build on current success
- Highlighting skills that could be developed further
- Recommending next steps or advanced techniques
- Finding opportunities for even better performance

If the context is entirely positive, frame constructive feedback around future growth opportunities."""
        else:
            tone_guidance = """
Focus on acknowledging achievements, highlighting strengths, and showing appreciation for good work."""
        
        prompt = f"""You are writing workplace feedback. Create a direct, professional feedback message.

Context: {context}
Tone: {tone}
Style: {style_instruction}

{tone_guidance}

Instructions:
1. Write direct feedback without greetings (no "Hi", "Dear", etc.)
2. Be specific about what happened and why it matters
3. Keep it 2-3 sentences, {style_instruction}
4. Always write in English regardless of input language
5. Make it genuine feedback based on the actual context provided
6. Match the {style} style - {style_instruction}
7. Handle context-tone mismatches intelligently - if asked for constructive feedback on positive context, focus on growth opportunities and next-level improvements

Generate the feedback:"""
        
        request_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        return self._invoke_model(request_payload)
    
    def _invoke_model(self, request_payload):
        """Invoke the Bedrock model with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                request_body = json.dumps(request_payload)
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                
                # Extract text from Claude response
                if "content" in response_body:
                    content_blocks = response_body["content"]
                    text = " ".join(
                        block["text"] for block in content_blocks 
                        if block.get("type") == "text" and block.get("text")
                    ).strip()
                    
                    return text
                else:
                    raise Exception("Unexpected response format")
                    
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                
                if error_code == 'ThrottlingException':
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise Exception(f"AWS Error: {error_code}")
                else:
                    time.sleep(1)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
        
        raise Exception("Max retries exceeded")

class ContextAnalyzer:
    def __init__(self, bedrock_client):
        self.client = bedrock_client
        self.model_id = os.getenv("AWS_BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
    
    def analyze_context_sentiment(self, context):
        """Use AI to analyze context sentiment"""
        if not self.client:
            # Fallback to simple keyword analysis
            return self._simple_sentiment_analysis(context)
        
        prompt = f"""Analyze the sentiment of this workplace context and respond with ONLY one word:

Context: "{context}"

Instructions:
- If the context describes good performance, achievements, success, or positive outcomes, respond: POSITIVE
- If the context describes problems, issues, failures, or areas needing improvement, respond: NEGATIVE  
- If the context is neutral or mixed, respond: NEUTRAL

Response (one word only):"""

        try:
            request_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            request_body = json.dumps(request_payload)
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            if "content" in response_body:
                content_blocks = response_body["content"]
                sentiment = " ".join(
                    block["text"] for block in content_blocks 
                    if block.get("type") == "text" and block.get("text")
                ).strip().upper()
                
                # Clean up response to get just the sentiment
                if "POSITIVE" in sentiment:
                    return "positive"
                elif "NEGATIVE" in sentiment:
                    return "negative"
                else:
                    return "neutral"
            
        except Exception as e:
            print(f"AI sentiment analysis failed: {e}")
            # Fallback to simple analysis
            return self._simple_sentiment_analysis(context)
    
    def _simple_sentiment_analysis(self, context):
        """Fallback simple sentiment analysis"""
        context_lower = context.lower()
        
        positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'impressed', 'successful', 'well done', 'achieved', 'delivered', 'completed', 'on time', 'quality work', 'effective', 'strong']
        negative_words = ['problem', 'issue', 'late', 'delayed', 'missed', 'failed', 'poor', 'bad', 'complaint', 'unhappy', 'disappointed', 'slow', 'mistake', 'error', 'weak', 'lacking']
        
        positive_count = sum(1 for word in positive_words if word in context_lower)
        negative_count = sum(1 for word in negative_words if word in context_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

def detect_context_tone_mismatch(context, tone, analyzer):
    """Detect if there's a mismatch between context sentiment and requested tone"""
    context_sentiment = analyzer.analyze_context_sentiment(context)
    
    # Detect mismatch
    mismatch = False
    
    if context_sentiment == 'positive' and tone == 'constructive':
        mismatch = True
    elif context_sentiment == 'negative' and tone == 'positive':
        mismatch = True
    
    return {
        'mismatch': mismatch,
        'context_sentiment': context_sentiment,
        'requested_tone': tone
    }

# Initialize AI generator and context analyzer
ai_generator = AIFeedbackGenerator()
context_analyzer = ContextAnalyzer(ai_generator.client)

# Validation class for feedback quality
class FeedbackValidator:
    """Simple feedback validator for database logging"""
    
    def validate_feedback(self, feedback: str, tone: str, context: str, style: str = "balanced") -> dict:
        """
        Basic validation of generated feedback
        Returns validation score and basic metrics
        """
        if not feedback or not feedback.strip():
            return {'overall_score': 0.0, 'is_valid': False}
        
        feedback_clean = feedback.strip()
        word_count = len(feedback_clean.split())
        sentence_count = len([s for s in feedback_clean.split('.') if s.strip()])
        
        # Basic scoring
        score = 1.0
        
        # Length check
        if word_count < 10:
            score -= 0.3
        elif word_count > 100:
            score -= 0.2
        
        # Sentence structure check
        if sentence_count < 1:
            score -= 0.4
        
        # Basic English check (simple heuristic)
        english_chars = sum(1 for c in feedback_clean if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in feedback_clean if c.isalpha())
        
        if total_chars > 0:
            english_ratio = english_chars / total_chars
            if english_ratio < 0.8:
                score -= 0.3
        
        score = max(0.0, min(1.0, score))
        
        return {
            'overall_score': score,
            'is_valid': score >= 0.5,
            'word_count': word_count,
            'sentence_count': sentence_count
        }

# Initialize validator
feedback_validator = FeedbackValidator()

@app.route('/')
def index():
    """Health check endpoint"""
    db_health = db_manager.health_check()
    return jsonify({
        'status': 'nGAGE AI Feedback Backend is running!',
        'ai_client': 'connected' if ai_generator.client else 'disconnected',
        'database': db_health['status'],
        'total_feedback_logged': db_health.get('total_records', 0)
    })

@app.route('/api/generate-feedback', methods=['POST'])
def generate_feedback_api():
    """API endpoint to generate feedback with database logging"""
    try:
        data = request.get_json()
        
        if not data or 'context' not in data or 'tone' not in data:
            return jsonify({'error': 'Missing context or tone'}), 400
        
        context = data['context'].strip()
        tone = data['tone'].lower()
        style = data.get('style', 'balanced').lower()
        
        if not context:
            return jsonify({'error': 'Context cannot be empty'}), 400
        
        if tone not in ['positive', 'constructive']:
            return jsonify({'error': 'Invalid tone. Must be positive or constructive'}), 400
        
        if style not in ['balanced', 'formal', 'casual', 'appreciative']:
            return jsonify({'error': 'Invalid style. Must be balanced, formal, casual, or appreciative'}), 400
        
        # Generate feedback using AI
        feedback = ai_generator.generate_feedback(context, tone, style)
        
        if not feedback:
            return jsonify({'error': 'Failed to generate feedback'}), 500
        
        # Validate feedback quality
        validation_result = feedback_validator.validate_feedback(feedback, tone, context, style)
        validation_score = validation_result.get('overall_score', 0)
        
        # Analyze context sentiment
        context_sentiment = context_analyzer.analyze_context_sentiment(context)
        
        # Get user info for logging
        user_ip = request.remote_addr
        session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
        
        # Log to database (non-blocking)
        try:
            print(f"🔍 Attempting to log feedback to database...")
            print(f"   Context: {context[:50]}...")
            print(f"   Tone: {tone}")
            print(f"   Style: {style}")
            print(f"   User IP: {user_ip}")
            print(f"   Session ID: {session_id}")
            
            success = db_manager.log_feedback(
                context=context,
                tone=tone,
                style=style,
                feedback=feedback,
                user_ip=user_ip,
                validation_score=validation_score,
                context_sentiment=context_sentiment,
                session_id=session_id
            )
            print(f"✅ Database logging result: {success}")
            
        except Exception as db_error:
            print(f"❌ Database logging failed (non-critical): {db_error}")
            import traceback
            traceback.print_exc()
        
        return jsonify({
            'feedback': feedback,
            'context': context,
            'tone': tone,
            'style': style,
            'session_id': session_id,
            'quality_score': validation_score
        })
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-context', methods=['POST'])
def analyze_context_api():
    """API endpoint to analyze context sentiment"""
    try:
        data = request.get_json()
        
        if not data or 'context' not in data or 'tone' not in data:
            return jsonify({'error': 'Missing context or tone'}), 400
        
        context = data['context'].strip()
        tone = data['tone'].lower()
        
        if not context:
            return jsonify({'error': 'Context cannot be empty'}), 400
        
        if tone not in ['positive', 'constructive']:
            return jsonify({'error': 'Invalid tone. Must be positive or constructive'}), 400
        
        # Analyze context for mismatch
        mismatch_info = detect_context_tone_mismatch(context, tone, context_analyzer)
        
        return jsonify({
            'mismatch': mismatch_info['mismatch'],
            'context_sentiment': mismatch_info['context_sentiment'],
            'requested_tone': mismatch_info['requested_tone']
        })
        
    except Exception as e:
        print(f"Error analyzing context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    db_health = db_manager.health_check()
    return jsonify({
        'status': 'healthy',
        'ai_client': 'connected' if ai_generator.client else 'disconnected',
        'database': db_health
    })

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data from database"""
    try:
        analytics = db_manager.get_analytics()
        if analytics is None:
            return jsonify({'error': 'Database not available'}), 503
        
        return jsonify(analytics)
        
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent-feedback')
def get_recent_feedback():
    """Get recent feedback entries (anonymized)"""
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(max(limit, 1), 50)  # Limit between 1-50
        
        recent = db_manager.get_recent_feedback(limit)
        if recent is None:
            return jsonify({'error': 'Database not available'}), 503
        
        return jsonify({
            'recent_feedback': recent,
            'count': len(recent)
        })
        
    except Exception as e:
        print(f"Error getting recent feedback: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting nGAGE AI Feedback Backend")
    print("=" * 50)
    print(f"Backend API: Running on Railway")
    print(f"Health: /api/health")
    print(f"Analytics: /api/analytics")
    print(f"Recent: /api/recent-feedback")
    
    # Database status
    db_health = db_manager.health_check()
    print(f"Database: {db_health['status']} - {db_health['message']}")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)