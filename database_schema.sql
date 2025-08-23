-- nGAGE AI Feedback Writer Database Schema
-- This file contains the database structure for logging feedback data

-- Main table to store feedback logs
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
    session_id VARCHAR(100),
    selected_attributes TEXT
);

-- Index for better query performance
CREATE INDEX IF NOT EXISTS idx_feedback_logs_timestamp ON feedback_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_logs_tone ON feedback_logs(selected_tone);
CREATE INDEX IF NOT EXISTS idx_feedback_logs_style ON feedback_logs(selected_style);

-- Optional: Create a view for analytics
CREATE OR REPLACE VIEW feedback_analytics AS
SELECT 
    DATE(timestamp) as date,
    selected_tone,
    selected_style,
    COUNT(*) as feedback_count,
    AVG(validation_score) as avg_quality_score,
    COUNT(DISTINCT user_ip) as unique_users
FROM feedback_logs 
GROUP BY DATE(timestamp), selected_tone, selected_style
ORDER BY date DESC;