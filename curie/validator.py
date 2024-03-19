import ast
import json
import re
from typing import List

from jsonschema import validate
from pydantic import BaseModel, ValidationError


class Quiz(BaseModel):
    question: str
    answer: str


class QuizCollection(BaseModel):
    collection: List[Quiz]


# serialize pydantic model into json schema
pydantic_schema = QuizCollection.schema_json()
json_schema = json.loads(pydantic_schema)


def extract_json_from_markdown(text):
    # Regular expression pattern to match simple dictionary-like structures
    dict_pattern = r"\{[^\{\}]*\}"

    # Find all substrings that look like dictionaries
    potential_dicts = re.findall(dict_pattern, text)

    extracted_dicts = []
    for d_str in potential_dicts:
        # Convert the string representation of dictionary to actual dictionary
        try:
            # Safe literal evaluation of dict string
            d = ast.literal_eval(d_str)
            if isinstance(d, dict):  # Ensure it is a dictionary
                extracted_dicts.append(d)
        except (ValueError, SyntaxError):
            # Skip if conversion fails
            continue

    return extracted_dicts


def validate_json_data(json_object):
    valid = False
    error_message = None
    result_json = None

    try:
        # Attempt to load JSON using json.loads
        try:
            result_json = json.loads(json_object)
        except json.decoder.JSONDecodeError:
            # If json.loads fails, try ast.literal_eval
            try:
                result_json = ast.literal_eval(json_object)
            except (SyntaxError, ValueError):
                try:
                    result_json = extract_json_from_markdown(json_object)
                except Exception as e:
                    error_message = f"JSON decoding error: {e}"
                    return valid, result_json, error_message

        # Return early if both json.loads and ast.literal_eval fail
        if result_json is None:
            error_message = "Failed to decode JSON data"
            return valid, result_json, error_message

        # Validate each item in the list against schema if it's a list
        if isinstance(result_json, list):
            for index, item in enumerate(result_json):
                try:
                    validate(instance=item, schema=json_schema)
                except ValidationError as e:
                    error_message = f"Validation failed for item {index + 1}: {e}"
                    break
        else:
            # Default to validation without list
            try:
                validate(instance=result_json, schema=json_schema)
            except ValidationError as e:
                error_message = f"Validation failed: {e}"

    except Exception as e:
        error_message = f"Error occurred: {e}"

    if error_message is None:
        valid = True

    return valid, result_json, error_message
