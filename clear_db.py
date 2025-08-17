#!/usr/bin/env python3
"""
Clear all entries from the nGAGE feedback database
"""

import sqlite3
import os

def clear_database():
    db_path = "ngage_local.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current count
        cursor.execute("SELECT COUNT(*) FROM feedback_logs")
        current_count = cursor.fetchone()[0]
        print(f"📊 Current entries in database: {current_count}")
        
        if current_count == 0:
            print("✅ Database is already empty!")
            return
        
        # Confirm deletion
        confirm = input(f"🗑️ Are you sure you want to delete all {current_count} entries? (yes/no): ")
        
        if confirm.lower() in ['yes', 'y']:
            # Delete all entries
            cursor.execute("DELETE FROM feedback_logs")
            conn.commit()
            
            # Verify deletion
            cursor.execute("SELECT COUNT(*) FROM feedback_logs")
            new_count = cursor.fetchone()[0]
            
            print(f"✅ Successfully deleted {current_count} entries!")
            print(f"📊 Remaining entries: {new_count}")
        else:
            print("❌ Deletion cancelled.")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    clear_database()
