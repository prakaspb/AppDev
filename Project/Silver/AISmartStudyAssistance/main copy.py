from database.progress_db import ProgressDB
from core.study_buddy import StudyBuddy

if __name__=='__main__':
    db=ProgressDB()
    sb=StudyBuddy(db)
    c=input('Concept: ')
    sb.study(c)


from ai.llm_factory import get_llm_engine
from database.progress_db import ProgressDB
from core.study_buddy import StudyBuddy
from utility.logger_config import setup_logger
import logging


def main():
    setup_logger()

    # ✅ USER-CONTROLLED FLAG
    USE_REAL_LLM = False   # 🔁 Change to True to use OpenAI

    logging.info(f"Using real LLM: {USE_REAL_LLM}")

    db = ProgressDB()
    llm_engine = get_llm_engine(USE_REAL_LLM)
    study_buddy = StudyBuddy(db, llm_engine)

    # rest of your existing logic...

