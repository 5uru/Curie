from mlx_lm import generate, load


def run_inference(prompt):
    config = {
        "verbose": True,
        "temp": 0.8,
        "max_tokens": 1500,
        "repetition_penalty": 1.1,
    }
    model_name = "jonathansuru/dolphin-2.1-mistral-7b-8bit"
    model, tokenizer = load(model_name, lazy=True)
    return generate(model, tokenizer, prompt=prompt, **config)


def generation(content, language, followings):
    prompt = f"""<|im_start|>system
You are a skilled flashcard creator. Your task is to generate a comprehensive set of questions and answers based on the provided text (CORPUS). It is crucial to maintain consistency in language by ensuring both the questions and answers are in the same language as the corpus. The content should focus on the specific topics requested by the user. The final output must be in the same language as the corpus. Always base the information on the corpus; never fabricate data. If there are no significant questions to be generated, you are not obligated to create them. However, if you do produce questions, they must adhere to the provided schema. Please provide the output in JSON format, following the schema provided below:


class Quiz(BaseModel):
    question: str
    answer: str

class QuizCollection(BaseModel):
    collection: List[Quiz]
Develop the output step by step, ensuring that you base your work solely on the corpus.

<|im_end|>
<|im_start|>user
Please generate a well-structured set of significant and relevant questions and answers based on the following corpus:
<|im_start|>corpus {content}
<|im_end|>
Ensure that the output is in JSON format, in the same language as {language.upper()}, and adheres to the provided schema.
<|im_end|>
<|im_start|>assistant
Step 1: Identify the key topics within the corpus based on the user's specifications.
Step 2: Select the most important and relevant information related to these topics.
Step 3: If there are significant questions to be generated, create a well-structured set of questions that effectively cover the chosen information. Otherwise, proceed to Step 5.
Step 4: Create concise and accurate answers to these questions using information from the corpus.
Step 5: Format the output as a JSON object, ensuring it is in the same language as {language.upper()} and follows the provided schema.

The following topics have been identified: {followings.upper()}. Based on these topics, here is a well-structured set of significant and relevant questions and answers derived from the corpus, prioritized according to their importance.
Based on the provided corpus, here is the output developed step by step, ensuring that the work is based solely on the corpus:

{{If there are significant questions:}}


{{
    "collection": [
        {{
    "question": "A relevant question based on the CORPUS, presented in the same language as {language.upper()}.",
            "answer": "A concise and accurate answer based on the CORPUS, provided in the same language as {language.upper()}.",
        }},
        ...
    ]
}}
{{If there are no significant questions:}}


{{
    "collection": []
}}
<|im_end|>
     """

    return run_inference(prompt)
