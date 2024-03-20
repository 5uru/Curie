import logging
from typing import List, Optional

import spacy
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from pke.unsupervised import TopicRank
from transformers import AutoTokenizer


def setup_logging():
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )


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
