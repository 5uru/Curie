from langchain.docstore.document import Document as LangchainDocument

from curie.llm_provider import generation
from curie.utils import get_topics, language_detection, split_documents
from curie.validator import validate_json_data


def quiz(content):
    lang = language_detection(content)
    topics = get_topics(content, lang, "fr_core_news_sm")

    corpus = [LangchainDocument(page_content=content.replace("\n\n", " "))]

    content_split = split_documents(1000, corpus)

    content_split = [doc.page_content for doc in content_split]

    all_quiz = []
    for doc in content_split:
        # Try to parse the JSON string
        doc_quiz = generation(doc, lang, " ,".join(topics))  # generate quiz
        validation, json_object, error_message = validate_json_data(doc_quiz)
        if validation:
            all_quiz.append(json_object)
        else:
            print(f"Validation failed: {error_message}")

    return all_quiz
