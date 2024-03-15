import ast

from langchain.docstore.document import Document as LangchainDocument

from curie.utils import language_detection, get_topics, split_documents, generation


def quiz(content):
    lang = language_detection(content)
    topics = get_topics(content, lang, "fr_core_news_sm")

    corpus = [LangchainDocument(page_content=content.replace("\n\n", " "))]

    content_split = split_documents(512, corpus)

    content_split = [doc.page_content for doc in content_split]

    all_quiz = []
    for doc in content_split:
        # Try to parse the JSON string
        doc_quiz = generation(doc, lang, " ,".join(topics))  # generate quiz
        print(type(doc_quiz))
        try:
            doc_quiz = eval(doc_quiz.replace("'", '"'))  # convert to json
            print(type(doc_quiz))
            quiz_collection = doc_quiz["collection"]  # get collection
            print(type(quiz_collection))
            quiz_collection = ast.literal_eval(quiz_collection)  # convert to list
            print(type(quiz_collection))
            all_quiz.extend(eval(_) for _ in quiz_collection)  # add to all_quiz
        except Exception:
            # If an error is raised, print an error message and the problematic string
            print("An error occurred while parsing the JSON string:")
            print(doc_quiz)
    return all_quiz
