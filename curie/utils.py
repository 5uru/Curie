from typing import Optional, List

import ollama
import spacy
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from pke.unsupervised import TopicRank
from transformers import AutoTokenizer


def language_detection(content):
    return detect(content)


def get_topics(content, language, spacy_model: Optional[str] = "en_core_web_sm"):
    # Load the SpaCy model
    nlp = spacy.load(spacy_model)

    # Create a TopicRank extractor
    extractor = TopicRank()

    # Load the content of the document
    extractor.load_document(
        content.replace("\n", " "),
        language=language,
        spacy_model=nlp,  # Pass the loaded SpaCy model
        normalization="stemming",
    )

    # Select the key phrase candidates
    extractor.candidate_selection()

    # Weight the candidates
    extractor.candidate_weighting()

    # The n-highest (5) scored candidates
    keyphrases = extractor.get_n_best(n=5, stemming=False)

    return [candidate for (candidate, _) in keyphrases]


EMBEDDING_MODEL_NAME = "thenlper/gte-small"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def generation(content, language, followings):
    prompt = f"""
     [DEFINITIONS]
- SYSTEM MESSAGE: Rules that the assistant must adhere to. THE OUTPUT MUST BE IN THE SAME LANGUAGE AS THE CORPUS. 
- USER GOAL: The desired outcome for the user.
- CORPUS: The base material for flashcard content.

[DIRECTIVES]
- Flashcard Creation: Assist in producing questions and answers for study based on the specified material.
- Language Consistency: Ensure all content matches the language of the provided material.
     OUTPUT MUST BE IN THE SAME LANGUAGE AS {language.upper()}. 
- THE QUESTION AND THE ANSWER MUST BE MAINLY RELATED TO THE FOLLOWING THEMES: {followings.upper()} 

[DETAILS]
- Content Requirements: Questions and answers must derive exclusively from the CORPUS and maintain thematic relevance.
- Language Specification: Maintain the same language as specified, applying to both questions and answers.

[OUTPUT TEMPLATE] - Format for Delivery: {{"collection": [{{"question"
    : "<Instruction for crafting questions aligning with user goals and language THE QUESTION MUST BE IN THE SAME 
     LANGUAGE AS THE CORPUS.>", 
      "answer": "<Guideline for providing answers based on the CORPUS, respecting user goals and specified themes. THE 
       ANSWER MUST BE IN THE SAME LANGUAGE AS THE CORPUS.>"}}]}}

[CORPUS] {content}


[THEMES] - Focus for Questions and Answers: Specified themes within the CORPUS. THE QUESTION AND THE ANSWER MUST BE 
 MAINLY RELATED TO THE FOLLOWING THEMES: {followings.upper()}

[NOTES]
- Ensure all flashcard content aligns with user instructions and system directives.
- Keep language consistent across questions and answers as per the CORPUS language.
DON'T INVENT THE DATA BASE ON THE CORPUS ALONE. 

Please output a collection between User and Assistant in the following format: {{ "collection": [ {{"question": 
 "followup question -Instruction for crafting questions aligning with user goals and language THE QUESTION MUST BE IN 
  THE SAME LANGUAGE AS THE CORPUS.", "answer": "the response - Guideline for providing answers based on the 
   CORPUS, respecting user goals and specified themes. THE ANSWER MUST BE IN THE SAME LANGUAGE AS THE CORPUS.'\n'"}}] 
    }}"""

    response = ollama.chat(
        model="dolphin-mistral", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
