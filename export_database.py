#!/usr/bin/env python3
"""
Database Export Tool for nGAGE AI Feedback Writer
Export database to CSV format
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

class DatabaseExporter:
    def __init__(self, db_path="ngage_local.db"):
        self.db_path = db_path
        self.export_dir = "exports"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create exports directory
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
    
    def get_database_data(self):
        """Get all data from database"""
        if not os.path.exists(self.db_path):
            print("❌ Database not found!")
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all feedback data
            query = """
                SELECT 
                    user_context as "User Input",
                    generated_feedback as "AI Generated Response"
                FROM feedback_logs 
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to read database: {e}")
            return None
    
    def export_to_csv(self):
        """Export database to CSV file"""
        print("📊 Exporting to CSV...")
        
        df = self.get_database_data()
        if df is None:
            return None
        
        # Data is already clean - just user input and AI response
        
        # Create filename
        csv_filename = f"{self.export_dir}/ngage_feedback_data_{self.timestamp}.csv"
        
        try:
            # Export to CSV
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            print(f"✅ CSV exported successfully!")
            print(f"📁 File: {os.path.abspath(csv_filename)}")
            print(f"📊 Records: {len(df)}")
            
            return csv_filename
            
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return None
    

    
    def export_analytics_summary(self):
        """Export analytics summary to text file"""
        print("📈 Exporting analytics summary...")
        
        df = self.get_database_data()
        if df is None:
            return None
        
        # Create filename
        summary_filename = f"{self.export_dir}/ngage_analytics_summary_{self.timestamp}.txt"
        
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("nGAGE AI Feedback Writer - Analytics Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
                
                # Basic stats
                f.write("📊 BASIC STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Feedback Generated: {len(df)}\n")
                f.write(f"Average Quality Score: {df['validation_score'].mean():.2f}\n")
                f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
                
                # Tone distribution
                f.write("🎯 TONE DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                tone_counts = df['selected_tone'].value_counts()
                for tone, count in tone_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{tone.title()}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Style distribution
                f.write("🎨 STYLE DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                style_counts = df['selected_style'].value_counts()
                for style, count in style_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{style.title()}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Sentiment distribution
                f.write("💭 CONTEXT SENTIMENT DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                sentiment_counts = df['context_sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{sentiment.title()}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Quality score distribution
                f.write("⭐ QUALITY SCORE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Minimum Score: {df['validation_score'].min():.2f}\n")
                f.write(f"Maximum Score: {df['validation_score'].max():.2f}\n")
                f.write(f"Average Score: {df['validation_score'].mean():.2f}\n")
                f.write(f"Median Score: {df['validation_score'].median():.2f}\n")
                
                # High quality feedback (score > 0.9)
                high_quality = df[df['validation_score'] > 0.9]
                f.write(f"High Quality Feedback (>0.9): {len(high_quality)} ({(len(high_quality)/len(df))*100:.1f}%)\n")
            
            print(f"✅ Analytics summary exported!")
            print(f"📁 File: {os.path.abspath(summary_filename)}")
            
            return summary_filename
            
        except Exception as e:
            print(f"❌ Analytics export failed: {e}")
            return None

def main():
    """Main export function"""
    print("🚀 nGAGE Database Export Tool")
    print("=" * 50)
    
    exporter = DatabaseExporter()
    
    print("Choose export format:")
    print("1. CSV only")
    print("2. Analytics summary only")
    print("3. Both CSV and Analytics")
    
    choice = input("Enter choice (1-3): ")
    
    exported_files = []
    
    if choice in ["1", "3"]:
        csv_file = exporter.export_to_csv()
        if csv_file:
            exported_files.append(csv_file)
    
    if choice in ["2", "3"]:
        summary_file = exporter.export_analytics_summary()
        if summary_file:
            exported_files.append(summary_file)
    
    if exported_files:
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 Files exported to: {os.path.abspath('exports')}")
        print("📋 Exported files:")
        for file in exported_files:
            print(f"   • {os.path.basename(file)}")
        
        # Open exports folder
        try:
            os.startfile(os.path.abspath("exports"))
            print("📂 Exports folder opened!")
        except:
            print("📂 Check the 'exports' folder for your files.")
    else:
        print("❌ No files were exported.")

if __name__ == "__main__":
    main()