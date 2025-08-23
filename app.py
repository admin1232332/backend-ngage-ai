
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

# Enhanced CORS configuration
CORS(app, 
     origins=['*'],  # Allow all origins for development
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
     supports_credentials=True)

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
    
    def generate_feedback(self, context, tone, style="balanced", selected_attributes=None):
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
        
        # Prepare attribute guidance if attributes are selected
        attribute_guidance = ""
        if selected_attributes and len(selected_attributes) > 0:
            attribute_list = ", ".join(selected_attributes)
            attribute_guidance = f"""
IMPORTANT: Focus your feedback specifically on these selected attributes: {attribute_list}
- Make sure your feedback directly relates to and mentions these specific qualities/skills
- Use these attributes as the framework for your feedback
- Be specific about how the person demonstrated these particular attributes"""
        
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

{attribute_guidance}

Instructions:
1. Write direct feedback without greetings (no "Hi", "Dear", etc.)
2. Be specific about what happened and why it matters
3. Keep it 2-3 sentences, {style_instruction}
4. Always write in English regardless of input language
5. Make it genuine feedback based on the actual context provided
6. Match the {style} style - {style_instruction}
7. Handle context-tone mismatches intelligently - if asked for constructive feedback on positive context, focus on growth opportunities and next-level improvements
8. If specific attributes are provided, structure your feedback around those attributes and mention them explicitly

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

# Predefined attributes organized by categories
FEEDBACK_ATTRIBUTES = {
    "Work Performance": [
        "Does Better than Expected", "Hardworking", "Performs Under Pressure", "Reliable", 
        "Resourceful", "Service-Oriented", "Team Player", "Technical Knowledge", 
        "Think on Their Feet", "Work Performance", "Problem Solver", "Availability", 
        "Domain Knowledge", "Result-Oriented", "Follows Processes", "Knowledge Sharing", 
        "Quality of Work", "Eye for Detail", "Understands & Follows Instructions", 
        "Beyond Call of Duty", "Analytical"
    ],
    "Attitude": [
        "Attitude", "Self-Organized", "Passion for Work", "Proactiveness", "Takes Ownership", 
        "Self-Motivated", "Confident & Bold", "Commitment", "Honest & Ethical", 
        "Learns from Mistakes", "Curious", "Flexible & Adaptable", "Creative & Innovative", "Loyal"
    ],
    "Communication Skills": [
        "Communicative", "Communication Skills", "Asks Questions", "Presentation Skills", 
        "Speaking Skills", "Storyteller", "Active Listener", "Thinks Before Speaking", 
        "Writing Skills", "Social Media Skills"
    ],
    "Leadership & Mgmt": [
        "Multitasking", "Takes Decisions", "Takes Initiatives", "Delegation", "Leadership & Mgmt", 
        "Manages Conflict", "Follows Up Well", "Manages Risk", "Strategic Thinker", 
        "Team Building", "Mentorship"
    ],
    "Culture": [
        "Rebel Ethos", "Agile Values", "Culture", "Embraces Vulnerability", "Fun Factor", 
        "Preaches Culture", "WOW Factor"
    ],
    "Emotional Intelligence": [
        "Empathy", "Respectful", "Emotionally Intelligent"
    ],
    "P&P Growth": [
        "Continuous Improvement", "Health & Fitness", "Learning", "P&P Growth", "Seeks Advice"
    ],
    "People Skills": [
        "Fair to Others", "Inspiration & Motivation", "Negotiator", "People Skills", 
        "Builds Relationships", "Appreciative", "Street Smart", "Energizing Others"
    ]
}

class AttributeRecommender:
    """AI-powered attribute recommendation system"""
    
    def __init__(self, bedrock_client):
        self.client = bedrock_client
        self.model_id = os.getenv("AWS_BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
        self.attributes_flat = []
        self.category_mapping = {}
        
        # Create flat list and category mapping
        for category, attributes in FEEDBACK_ATTRIBUTES.items():
            for attr in attributes:
                self.attributes_flat.append(attr)
                self.category_mapping[attr] = category
    
    def recommend_attributes(self, context: str, max_recommendations: int = 5) -> dict:
        """
        Recommend relevant attributes based on context using AI
        
        Args:
            context: User's input context
            max_recommendations: Maximum number of attributes to recommend
            
        Returns:
            Dict with recommended attributes and categories
        """
        print(f"🎯 AttributeRecommender.recommend_attributes called")
        print(f"📝 Context: {context[:100]}...")
        print(f"🔢 Max recommendations: {max_recommendations}")
        print(f"🤖 Client available: {self.client is not None}")
        print(f"📊 Total attributes available: {len(self.attributes_flat)}")
        
        # User has confirmed context is complete, proceed with recommendations
        
        if not self.client:
            print("⚠️ No AI client available, using fallback")
            return self._fallback_recommendation(context, max_recommendations)
        
        # Create attribute list for AI prompt
        attributes_text = "\n".join([f"- {attr}" for attr in self.attributes_flat])
        
        prompt = f"""Analyze this workplace context and recommend the most relevant feedback attributes from the provided list.

Context: "{context}"

Available Attributes:
{attributes_text}

Instructions:
1. Analyze the context to understand what skills, behaviors, or qualities are being demonstrated
2. Select the {max_recommendations} most relevant attributes from the list above
3. Focus on attributes that directly relate to what's described in the context
4. Consider both positive and areas for improvement mentioned
5. Respond with ONLY the exact attribute names from the list, one per line
6. Do not add explanations or modify the attribute names

Recommended attributes:"""

        try:
            print("🚀 Making AI request for attribute recommendations...")
            request_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.3,
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
            print(f"✅ AI response received: {response_body}")
            
            if "content" in response_body:
                content_blocks = response_body["content"]
                ai_response = " ".join(
                    block["text"] for block in content_blocks 
                    if block.get("type") == "text" and block.get("text")
                ).strip()
                
                print(f"📝 AI raw response: {ai_response}")
                result = self._parse_ai_recommendations(ai_response, max_recommendations)
                print(f"🎯 Parsed recommendations: {result}")
                return result
            else:
                print("❌ No content in AI response")
                return self._fallback_recommendation(context, max_recommendations)
            
        except Exception as e:
            print(f"❌ AI attribute recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to keyword-based recommendation...")
            return self._fallback_recommendation(context, max_recommendations)
    
    def _parse_ai_recommendations(self, ai_response: str, max_recommendations: int) -> dict:
        """Parse AI response and validate attribute names"""
        lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
        recommended_attributes = []
        
        for line in lines:
            # Clean up the line (remove bullets, numbers, etc.)
            clean_line = line.replace('-', '').replace('•', '').replace('*', '').strip()
            
            # Find exact matches in our attribute list
            for attr in self.attributes_flat:
                if clean_line.lower() == attr.lower() or attr.lower() in clean_line.lower():
                    if attr not in recommended_attributes:
                        recommended_attributes.append(attr)
                        break
            
            if len(recommended_attributes) >= max_recommendations:
                break
        
        # If we don't have enough recommendations, add fallback
        if len(recommended_attributes) < max_recommendations:
            fallback = self._fallback_recommendation("", max_recommendations - len(recommended_attributes))
            for attr in fallback['recommended_attributes']:
                if attr not in recommended_attributes:
                    recommended_attributes.append(attr)
        
        # Group by categories
        categories = {}
        for attr in recommended_attributes:
            category = self.category_mapping.get(attr, "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(attr)
        
        return {
            'recommended_attributes': recommended_attributes,
            'categories': categories,
            'total_count': len(recommended_attributes)
        }
    
    def _fallback_recommendation(self, context: str, max_recommendations: int) -> dict:
        """Fallback recommendation using keyword matching"""
        print(f"🔄 Using fallback recommendation for context: {context[:50]}...")
        context_lower = context.lower()
        
        # Keyword-based matching
        keyword_mapping = {
            'deadline': ['Reliable', 'Work Performance', 'Follows Processes'],
            'communication': ['Communication Skills', 'Communicative', 'Active Listener'],
            'team': ['Team Player', 'Team Building', 'Builds Relationships'],
            'presentation': ['Presentation Skills', 'Communication Skills', 'Confident & Bold'],
            'problem': ['Problem Solver', 'Analytical', 'Think on Their Feet'],
            'quality': ['Quality of Work', 'Eye for Detail', 'Technical Knowledge'],
            'leadership': ['Leadership & Mgmt', 'Takes Initiatives', 'Takes Decisions'],
            'learning': ['Learning', 'Curious', 'Continuous Improvement'],
            'help': ['Team Player', 'Mentorship', 'Knowledge Sharing'],
            'client': ['Service-Oriented', 'Communication Skills', 'Result-Oriented']
        }
        
        recommended = []
        for keyword, attributes in keyword_mapping.items():
            if keyword in context_lower:
                for attr in attributes:
                    if attr not in recommended and len(recommended) < max_recommendations:
                        recommended.append(attr)
        
        # Fill with default attributes if needed
        default_attributes = ['Work Performance', 'Communication Skills', 'Team Player', 'Attitude', 'Quality of Work']
        for attr in default_attributes:
            if attr not in recommended and len(recommended) < max_recommendations:
                recommended.append(attr)
        
        # Group by categories
        categories = {}
        for attr in recommended:
            category = self.category_mapping.get(attr, "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(attr)
        
        return {
            'recommended_attributes': recommended,
            'categories': categories,
            'total_count': len(recommended)
        }

# Initialize attribute recommender
attribute_recommender = AttributeRecommender(ai_generator.client)

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
        
        context = data.get('context')
        if context is None:
            return jsonify({'error': 'Context cannot be null'}), 400
            
        context = str(context).strip()
        tone = data['tone'].lower()
        style = data.get('style', 'balanced').lower()
        selected_attributes = data.get('selected_attributes', [])
        
        if not context:
            return jsonify({'error': 'Context cannot be empty'}), 400
            
        # Validate context length and quality
        word_count = len(context.split())
        

        # Email and message patterns that indicate non-feedback content
        email_patterns = [
            'email about', 'letter about', 'message about',
            'write an email', 'write a letter', 'write a message',
            'draft an email', 'draft a letter',
            'compose an email', 'compose a letter',
            'dear', 'subject:', 'regarding:', 'ref:', 're:', 'fw:', 'fwd:'
        ]
        
        
        context_lower = context.lower()
        words = context_lower.split()
        first_few_words = ' '.join(words[:4]).lower()  # Get first 4 words
        
        # Check if this is a feedback request
        is_feedback_request = 'feedback' in context_lower or (
            'write' in first_few_words and 'feedback' in context_lower
        )
        
        # First check for email/letter writing requests (but allow feedback requests)
        if not is_feedback_request and any(pattern in context_lower for pattern in email_patterns):
            return jsonify({
                'error': 'I am not designed to write emails or letters. I can only provide feedback on work performance.',
                'details': 'This tool is specifically for performance feedback. For emails, please use an email writing tool instead.'
            }), 400
            
            
            
        # Check for extremely short or single-word responses that got through other filters
        if len(words) < 3:
            return jsonify({
                'error': 'Please provide more detailed context about the work or performance.',
                'details': 'Single words or very short phrases are not sufficient for meaningful feedback.'
            }), 400
            
        if word_count < 5:  # Minimum 5 words for meaningful context
            return jsonify({
                'error': 'Please provide more context about the situation or performance you want feedback on (at least 5 words).',
                'details': f'Current input: {word_count} words. Minimum required: 5 words.'
            }), 400
        
        # Map common tone values to supported ones
        tone_mapping = {
            'professional': 'constructive',
            'encouraging': 'positive',
            'supportive': 'positive',
            'formal': 'constructive'
        }
        
        if tone in tone_mapping:
            tone = tone_mapping[tone]
        
        if tone not in ['positive', 'constructive']:
            return jsonify({'error': 'Invalid tone. Must be positive or constructive'}), 400
        
        if style not in ['balanced', 'formal', 'casual', 'appreciative']:
            return jsonify({'error': 'Invalid style. Must be balanced, formal, casual, or appreciative'}), 400
        
        # Generate feedback using AI
        feedback = ai_generator.generate_feedback(context, tone, style, selected_attributes)
        
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
                session_id=session_id,
                selected_attributes=selected_attributes
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
            'quality_score': validation_score,
            'selected_attributes': selected_attributes if selected_attributes else []
        })
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-context', methods=['POST'])
def analyze_context_api():
    """API endpoint to analyze context sentiment"""
    try:
        data = request.get_json()
        
        if not data or 'context' not in data:
            return jsonify({'error': 'Missing context'}), 400
        
        context = data.get('context')
        if context is None:
            return jsonify({'error': 'Context cannot be null'}), 400
            
        context = str(context).strip()
        tone = data.get('tone', 'constructive').lower()  # Default tone if not provided
        
        if not context:
            return jsonify({'error': 'Context cannot be empty'}), 400
        
        # Map common tone values to supported ones
        tone_mapping = {
            'professional': 'constructive',
            'encouraging': 'positive',
            'supportive': 'positive',
            'formal': 'constructive'
        }
        
        if tone in tone_mapping:
            tone = tone_mapping[tone]
        
        if tone not in ['positive', 'constructive']:
            tone = 'constructive'  # Default to constructive if invalid
        
        # Analyze context for mismatch
        mismatch_info = detect_context_tone_mismatch(context, tone, context_analyzer)
        
        # Also analyze for basic sentiment and themes
        try:
            basic_analysis = context_analyzer.analyze_context_sentiment(context)
            if isinstance(basic_analysis, dict):
                key_themes = basic_analysis.get('key_themes', [])
            else:
                key_themes = []
        except:
            key_themes = []
        
        return jsonify({
            'sentiment': mismatch_info.get('context_sentiment', 'neutral'),
            'key_themes': key_themes,
            'mismatch': mismatch_info['mismatch'],
            'context_sentiment': mismatch_info['context_sentiment'],
            'requested_tone': mismatch_info['requested_tone']
        })
        
    except Exception as e:
        print(f"Error analyzing context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        return response
        
    db_health = db_manager.health_check()
    response = jsonify({
        'status': 'healthy',
        'ai_client': 'connected' if ai_generator.client else 'disconnected',
        'database': db_health
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

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

@app.route('/api/recommend-attributes', methods=['POST', 'OPTIONS'])
def recommend_attributes_api():
    """API endpoint to recommend relevant attributes based on context"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        print("🎯 Attribute recommendation API called")
        data = request.get_json()
        print(f"📝 Request data: {data}")
        
        if not data or 'context' not in data:
            print("❌ Missing context in request")
            return jsonify({'error': 'Missing context'}), 400
        
        context = data['context'].strip()
        max_recommendations = data.get('max_recommendations', 5)
        
        print(f"📋 Context: {context[:100]}...")
        print(f"🔢 Max recommendations: {max_recommendations}")
        
        if not context:
            print("❌ Empty context provided")
            return jsonify({'error': 'Context cannot be empty'}), 400
        
        # Basic validation - user has already confirmed context is complete
        if len(context) < 10:
            return jsonify({'error': 'Context too short for meaningful suggestions'}), 400
        
        # Validate max_recommendations
        max_recommendations = min(max(max_recommendations, 1), 10)  # Between 1-10
        
        # Check if attribute_recommender is properly initialized
        if not attribute_recommender:
            print("❌ Attribute recommender not initialized")
            return jsonify({'error': 'Attribute recommender not available'}), 500
        
        print("🤖 Getting AI recommendations...")
        # Get AI recommendations
        recommendations = attribute_recommender.recommend_attributes(context, max_recommendations)
        print(f"✅ Recommendations generated: {recommendations}")
        
        return jsonify({
            'context': context,
            'recommendations': recommendations,
            'all_attributes': FEEDBACK_ATTRIBUTES  # Include full list for reference
        })
        
    except Exception as e:
        print(f"❌ Error recommending attributes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/attributes')
def get_all_attributes():
    """Get all available feedback attributes organized by categories"""
    try:
        return jsonify({
            'attributes': FEEDBACK_ATTRIBUTES,
            'total_count': sum(len(attrs) for attrs in FEEDBACK_ATTRIBUTES.values())
        })
        
    except Exception as e:
        print(f"Error getting attributes: {e}")
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