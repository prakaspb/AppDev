from rag import get_rag_chain


def ask(chain, question):
    result = chain.run(question)
    print("\nAnswer:\n")
    print(result)
    print("-" * 70)



def policy_question_flow(chain):
    question = input("\nEnter your policy question:\n> ")
    ask(chain, question)


def claim_precheck_flow(chain):
    print("\nEnter claim details (be descriptive):")
    claim_details = input("""
Policy duration (months):
Hospitalization (Yes/No):
Reason for hospitalization / treatment:
Pre-existing condition (Yes/No):
""")

    prompt = f"""
Evaluate the following claim scenario using the policy.
Identify:
- Eligibility
- Waiting period issues
- Exclusions
- Risk of rejection (if any)

Claim details:
{claim_details}
"""
    ask(chain, prompt)


def main():
    print("\ Policy & Claims Copilot (Mistral + RAG)")
    print("------------------------------------------------\n")

    chain = get_rag_chain()

    while True:
        print("\nChoose an option:")
        print("1. Ask a policy coverage question")
        print("2. Run a claim pre-check")
        print("3. Exit")

        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            policy_question_flow(chain)
        elif choice == "2":
            claim_precheck_flow(chain)
        elif choice == "3":
            print("\n✅ Exiting Copilot. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()