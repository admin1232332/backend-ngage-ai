# Database Setup Instructions for nGAGE AI Feedback Writer

## Railway PostgreSQL Setup

### Step 1: Add PostgreSQL Service
1. Go to your Railway project dashboard
2. Click "Add Service" or "+"
3. Select "Database" → "PostgreSQL"
4. Railway will automatically create the database and set environment variables

### Step 2: Environment Variables
Railway automatically sets these variables:
- `DATABASE_URL` - Complete PostgreSQL connection string
- `PGHOST` - Database host
- `PGPORT` - Database port (usually 5432)
- `PGUSER` - Database username
- `PGPASSWORD` - Database password
- `PGDATABASE` - Database name

### Step 3: Verify Connection
Your app will automatically:
- Connect to the database on startup
- Create required tables if they don't exist
- Show connection status in logs

### Step 4: Check Database Health
Visit these endpoints to verify database functionality:
- `GET /api/health` - Shows database connection status
- `GET /api/analytics` - Shows usage analytics
- `GET /api/recent-feedback` - Shows recent feedback entries

## Database Schema

### Tables Created Automatically:
- `feedback_logs` - Main table storing all feedback data
- Indexes for performance optimization
- Analytics view for reporting

### Data Stored:
- User context/notes (truncated to 1000 chars)
- Selected tone and style
- Generated feedback (truncated to 2000 chars)
- Timestamp
- User IP (optional, for analytics)
- Validation score (AI quality assessment)
- Context sentiment (positive/negative/neutral)
- Session ID (for tracking user sessions)

## Privacy & Security

### Data Protection:
- User context is truncated to prevent excessive storage
- IP addresses are optional and can be disabled
- No personally identifiable information is stored
- Session IDs are randomly generated UUIDs

### Data Retention:
- Consider implementing data retention policies
- Add cleanup jobs for old data if needed
- Monitor database size and usage

## Troubleshooting

### Common Issues:
1. **Connection Failed**: Check if DATABASE_URL is set correctly
2. **Table Creation Failed**: Ensure database user has CREATE permissions
3. **Logging Failed**: App continues working even if database logging fails

### Debug Commands:
```bash
# Check environment variables
echo $DATABASE_URL

# Test database connection (if you have psql installed)
psql $DATABASE_URL -c "SELECT version();"

# View recent logs
railway logs
```

## Analytics Queries

### Useful SQL Queries:
```sql
-- Total feedback by tone
SELECT selected_tone, COUNT(*) 
FROM feedback_logs 
GROUP BY selected_tone;

-- Daily usage
SELECT DATE(timestamp), COUNT(*) 
FROM feedback_logs 
GROUP BY DATE(timestamp) 
ORDER BY DATE(timestamp) DESC;

-- Average quality scores
SELECT AVG(validation_score) 
FROM feedback_logs 
WHERE validation_score IS NOT NULL;

-- Most popular styles
SELECT selected_style, COUNT(*) 
FROM feedback_logs 
GROUP BY selected_style 
ORDER BY COUNT(*) DESC;
```

## Cost Estimation

### Railway PostgreSQL Pricing:
- **Hobby Plan**: $5/month for 1GB storage
- **Pro Plan**: $10/month for 8GB storage
- Additional storage: $2/GB/month

### Expected Usage:
- Each feedback entry: ~1-2KB
- 1000 feedback entries: ~1-2MB
- Should easily fit in Hobby plan for most use cases

## Backup & Recovery

### Automatic Backups:
- Railway provides automatic daily backups
- Point-in-time recovery available
- Backups retained for 7 days (Hobby) or 30 days (Pro)

### Manual Backup:
```bash
# Export data
pg_dump $DATABASE_URL > backup.sql

# Import data
psql $DATABASE_URL < backup.sql
```