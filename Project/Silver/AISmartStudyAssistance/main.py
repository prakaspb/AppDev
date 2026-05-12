from database.progress_db import ProgressDB
from core.study_buddy import StudyBuddy

import logging
from utility.logger_config import setup_logger
from ai.llm_factory import get_llm_engine




def main():
    print("\n📚 Welcome to AI Smart Study Buddy")
        
    ut = setup_logger()
    logging.info("AI Smart Study Buddy started")
    
    # ✅ USER-CONTROLLED FLAG
    USE_REAL_LLM = False   # 🔁 Change to True to use OpenAI

    logging.info(f"Using real LLM: {USE_REAL_LLM}")



    # Initialize database and study buddy
    
    db = ProgressDB()
    llm_engine = get_llm_engine(USE_REAL_LLM)
    study_buddy = StudyBuddy(db, llm_engine)


    # Step 1: Ingest study material
    file_path = input("\nEnter PDF path: ").strip()
    logging.info(f"PDF provided: {file_path}")

    try:
        concepts = study_buddy.ingest_material(file_path)
        logging.info(f"Extracted concepts: {concepts}")
    except Exception as e:
        print(f"\n❌ Failed to read or analyze the file: {e}")
        logging.exception("Failed during material ingestion")
        return

    if not concepts:
        print("\n⚠️ No concepts could be extracted.")
        return

    # Step 2: Display extracted concepts
    print("\n📌 Extracted concepts:")
    for idx, concept in enumerate(concepts, start=1):
        print(f"{idx}. {concept}")

    # Step 3: Interactive study loop
    while True:
        print("\nOptions:")
        print("1. Study a concept")
        print("2. View progress")
        print("3. Exit")

        choice = input("Select an option: ").strip()
        logging.info(f"User selected option: {choice}")

        if choice == "1":
            concept = input("Enter concept name (or number): ").strip()
            logging.info(f"Studying concept: {concept}")

            # Allow numeric or text selection
            if concept.isdigit():
                index = int(concept) - 1
                if 0 <= index < len(concepts):
                    concept = concepts[index]
                else:
                    print("Invalid concept number.")
                    continue

            study_buddy.study_concept(concept)

        elif choice == "2":
            db.show_progress()
            #logginh.info("Displayed progress to user")
            logging.info("Displaying progress")

        elif choice == "3":
            print("\n✅ Exiting AI Smart Study Buddy. Happy learning!")
            logging.info("Application exited by user")
            break

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()