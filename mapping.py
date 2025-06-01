import os
import json
import re
from google import genai

client = genai.Client(api_key="API_KEY") 

pku_categories = [
    "Psychological Harm",
    "Sexual Content",
    "Privacy Violation",
    "Cybercrime",
    "Insulting Behavior",
    "Misinformation",
    "Endangering Public Health",
    "Physical Harm",
    "Financial Fraud",
    "Terrorism",
    "Hate Speech",
    "Self‑Harm Encouragement",
    "Drug Advice",
    "Violent Extremism",
    "Underage Substance Use",
    "Economic Crime",
    "Gambling Advice",
    "Animal Cruelty",
    "Other"
]

minorbench_categories = [
    "Sexual",
    "Profanities",
    "Hateful",
    "Danger",
    "Self‑harm",
    "Substance use"
]


def map_with_gemini(pku_list, child_list):

    system_prompt = (
        "You are an expert in content safety for children. "
        "You have GENERAL HARM categories that may appear in user‑generated content, "
        "and you need to map each one to a CHILD‑HARM category reflecting risk to minors."
    )
    user_prompt = (
        "GENERAL HARM categories:\n" +
        json.dumps(pku_list, indent=2) +
        "\n\nCHILD‑HARM categories:\n" +
        json.dumps(child_list, indent=2) +
        "\n\nFor each GENERAL category, choose exactly one CHILD‑HARM category.  "
        "Output ONLY a JSON object whose keys are the exact GENERAL names "
        "and whose values are the exact CHILD‑HARM names.  No extra text."
    )


    combined_prompt = system_prompt + "\n\n" + user_prompt


    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=combined_prompt
    )


    text = response.text.strip()

    try:
        mapping = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        mapping = json.loads(m.group(0))

    missing = set(pku_list) - set(mapping.keys())
    if missing:
        raise ValueError(f"Gemini did not map these PKU categories: {missing}")

    return mapping

if __name__ == "__main__":
    mapping = map_with_gemini(pku_categories, minorbench_categories)
    print(json.dumps(mapping, indent=2))
