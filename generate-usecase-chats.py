import random

import ollama

system_prompts_file = "data/systemprompts.txt"
usecases_file = "data/usecase_archetypes.txt"
out_file = "data/usecase_conversations_sharegpt.jsonl"

f = open(out_file, "a", encoding="utf-8")

system_prompts = open(system_prompts_file, "r").readlines()
random.shuffle(system_prompts)
usecases = open(usecases_file, "r").readlines()
random.shuffle(usecases)

while True:
    usecase = random.choice(usecases)
    system_prompt = random.choice(system_prompts)

    prompt = """
     [DEFINITIONS]
- SYSTEM MESSAGE: Rules that the assistant must adhere to.
- USER GOAL: The desired outcome for the user.
- CORPUS: The base material for flashcard content.

[DIRECTIVES]
- Flashcard Creation: Assist in producing questions and answers for study based on the specified material.
- Language Consistency: Ensure all content matches the language of the provided material.

[DETAILS]
- Content Requirements: Questions and answers must derive exclusively from the CORPUS and maintain thematic relevance.
- Language Specification: Maintain the same language as specified, applying to both questions and answers.

[OUTPUT TEMPLATE] - Format for Delivery: {"collection": [{"question": "<Instruction for crafting questions aligning 
 with user goals and language>", "answer": "<Guideline for providing answers based on the CORPUS, respecting user 
  goals and specified themes.>"}]}

[THEMES]
- Focus for Questions and Answers: Specified themes within the CORPUS.

[NOTES]
- Ensure all flashcard content aligns with user instructions and system directives.
- Keep language consistent across questions and answers as per the CORPUS language.
DON'T INVENT THE DATA BASE ON THE CORPUS ALONE. 

     """
    print("SYSTEM MESSAGE", system_prompt)
    print("USER GOAL", usecase)
    response = ollama.chat(
        model="dolphin-mistral", messages=[{"role": "user", "content": prompt}]
    )
    print(response["message"]["content"])
