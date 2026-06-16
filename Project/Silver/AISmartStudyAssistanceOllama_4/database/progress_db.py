import sqlite3
from datetime import date
import logging
import sqlite3
import logging
from datetime import date
from scheduler.spaced_repetition import get_review_schedule

class ProgressDB:
    def __init__(self):
        self.conn = sqlite3.connect("study_buddy_ppb13.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                concept TEXT PRIMARY KEY,
                mastery REAL,
                last_review DATE
            )
        """)

    def update(self, concept, mastery):
        self.conn.execute(
            "INSERT OR REPLACE INTO progress VALUES (?, ?, ?)",
            (concept, mastery, date.today())
        )
        self.conn.commit()

    def get_mastery(self, concept):
        cur = self.conn.cursor()
        cur.execute("SELECT mastery FROM progress WHERE concept = ?", (concept,))
        row = cur.fetchone()
        return row[0] if row else 0.0

    # Method to display all progress records in a user-friendly format
    def show_progress1(self):
        cur = self.conn.cursor()
        cur.execute("SELECT concept, mastery, last_review FROM progress")
        rows = cur.fetchall()

        if not rows:
            print("\n📭 No learning progress recorded yet.")
            logging.info("Progress requested: no records found.")
            return

        print("\n📊 Learning Progress")
        logging.info("Learning progress displayed:")
        for concept, mastery, last_review in rows:
            status = self._mastery_status(mastery)
            print(
                f"- {concept} | Mastery: {mastery:.2f} ({status}) | Last Reviewed: {last_review}"
            )
        
        logging.info(
            f"- {concept} | Mastery: {mastery:.2f} ({status}) | Last Reviewed: {last_review}"
        )

    

    
    def show_progress(self):
        cur = self.conn.cursor()
        cur.execute("SELECT concept, mastery, last_review FROM progress")
        rows = cur.fetchall()

        if not rows:
            print("\n📭 No learning progress recorded yet.")
            logging.info("Progress requested: no records found.")
            return

        print("\n📊 Learning Progress & Review Schedule")
        logging.info("Learning progress displayed:")

        for concept, mastery, last_review in rows:
            status = self._mastery_status(mastery)

            days, next_review = get_review_schedule(
                mastery,
                date.fromisoformat(last_review)
            )

            line = (
                f"- {concept} | Mastery: {mastery:.2f} ({status}) | "
                f"Last Reviewed: {last_review} | "
                f"Next Review: {next_review} (in {days} days)"
            )

            print(line)
            logging.info(line)

    # ✅ INTERNAL HELPER
    def _mastery_status(self, mastery):
        if mastery >= 0.8:
            return "Strong"
        elif mastery >= 0.5:
            return "Improving"
        return "Needs Review"

