__all__ = [
    "check_facts",
    "extract_facts"
]

section_to_fact_definition = {
    "Duration of Study Involvement": [
        {
            "fact_name": "Study total duration",
            "fact_type": "numerical",
            "fact_description": "The total duration of the study in days, weeks, months, etc."
        },
        {
            "fact_name": "Active participation duration",
            "fact_type": "numerical",
            "fact_description": "The duration of active participation by each participant in days, weeks, months, etc."
        },
        {
            "fact_name": "Follow-up period",
            "fact_type": "numerical",
            "fact_description": "The time duration of the follow-up period in days, weeks, months, etc."
        },
        {
            "fact_name": "Washout period",
            "fact_type": "numerical",
            "fact_description": "The time duration of the washout period in days, weeks, months, etc."
        }
    ],
    "Procedures": [
        {
            "fact_name": "Pre-treatment test procedures",
            "fact_type": "list[str]",
            "fact_description": "List of test procedures before the treatment procedures. For each, return the name, frequency, and duration."
        },
        {
            "fact_name": "Treatment procedures",
            "fact_type": "list[str]",
            "fact_description": "List of treatment procedures. For each, return the name, frequency, and duration."
        },
        {
            "fact_name": "Follow-up procedures",
            "fact_type": "list[str]",
            "fact_description": "List of follow-up procedures. For each, return the name, frequency, and duration."
        },
        {
            "fact_name": "Invastive procedures",
            "fact_type": "list[str]",
            "fact_description": "List of names of treatments/tests that are invasive."
        },
        {
            "fact_name": "Specimen collection",
            "fact_type": "list[str]",
            "fact_description": "List of specimens to be obtained, the estimated frequency, volume, or size."
        },
        {
            "fact_name": "Contraception",
            "fact_type": "bool",
            "fact_description": "Whether contraception is needed."
        },
        {
            "fact_name": "Genetic testing",
            "fact_type": "bool",
            "fact_description": "Whether future research on specimens will include genetic testing."
        },
        {
            "fact_name": "Whole genome sequencing",
            "fact_type": "bool",
            "fact_description": "Whether whole genome sequencing will occur."
        },
        {
            "fact_name": "Sharing research results",
            "fact_type": "bool",
            "fact_description": "Whether the results of the study will be shared with the participant."
        },
        {
            "fact_name": "MRI",
            "fact_type": "bool",
            "fact_description": "Whether MRI will be used in the study."
        },
        {
            "fact_name": "Unapproved devices",
            "fact_type": "list[str]",
            "fact_description": "List of names of radio frequency coil, device, or software that has not been approved by the FDA."
        }
    ],
    "Possible Risks, Discomforts, and Inconveniences": [
        {
            "fact_name": "Treatment side effects",
            "fact_type": "list[str]",
            "fact_description": "List of the name of side effects that can be caused by the treatments."
        },
        {
            "fact_name": "Test side effects",
            "fact_type": "list[str]",
            "fact_description": "List of the name of side effects that can be caused by the tests conducted in the study."
        },
        {
            "fact_name": "Discontinuing medications",
            "fact_type": "list[str]",
            "fact_description": "List of the name of risks associated with discontinuing medications."
        }
    ]
}

PROMPT_TEMPLATE = """
Your task is to extract the key information items from the given content. 
Do not extract any information item that is not explicitly mentioned in the content.
The extracted value should be as concise as possible.

Content:
{content}

Key items to extract:
"""

FACT_DEFINITION_TEMPLATE = "`{fact_name}` (type: `{fact_type}`): {fact_description}"

OUTPUT_TEMPLATE = """Your output should be a JSON object with the following format:
```json
{{
    "extracted": [
        {{ 
            "name": "fact_name_1",
            "value": "fact value 1"
        }}, \\ informatino item 1
        {{
            "name": "fact_name_2",
            "value": "fact value 2"
        }}, \\ informatino item 2
        ... \\ more items if needed
    ]
}}
```

Output:
```json
"""

import pdb
import pandas as pd
import json
from benchmark.llm import call_llm_json_output


def _build_fact_definition(facts):
    """Build the fact definition string from the given facts.
    """
    fact_definitions = []
    for fact in facts:
        fact_definitions.append(FACT_DEFINITION_TEMPLATE.format(**fact))
    return "-" + "\n\n".join(fact_definitions)

def get_prompt_for_fact_extraction(section_title):
    """Get the prompt for extracting facts from the given section."""
    facts = section_to_fact_definition.get(section_title, [])
    fact_definitions = _build_fact_definition(facts)
    prompt_template = PROMPT_TEMPLATE + fact_definitions + "\n\n" + OUTPUT_TEMPLATE
    return prompt_template

def check_facts(content, facts, section_title, llm):
    """Check if the given content contains the specified facts."""
    # stringify the facts
    fact_strs = [f"<{i}>{fact}</{i}>" for i, fact in enumerate(facts)]
    fact_strs = "\n".join(fact_strs)
    prompt_template = {
        "Possible Risks, Discomforts, and Inconveniences": """

Task: For each #FACT, check if the any of the risks/discomforts/inconveniences described are present in the #CONTENT.

#FACTs:
{fact_strs}

#CONTENT:
{content}

Return a JSON object with the key the index of the fact and the value as a boolean indicating if the fact is present in the content, e.g.,
{{
    "0": true,
    "1": false,
    ...
}}
""",
    "Procedures": """

Task: For each #FACT, check if the main procedure described are present in the #CONTENT.

#FACTs:
{fact_strs}

#CONTENT:
{content}

Return a JSON object with the key the index of the fact and the value as a boolean indicating if the fact is present in the content, e.g.,
{{
    "0": true,
    "1": false,
    ...
}}

    """,
    "Duration of Study Involvement": """
Task: For each #FACT, check if the main duration described is present in the #CONTENT.

#FACTs:
{fact_strs}

#CONTENT:
{content}

Return a JSON object with the key the index of the fact and the value as a boolean indicating if the fact is present in the content, e.g.,
{{
    "0": true,
    "1": false,
    ...
}}
"""
    }
    prompt_template = prompt_template[section_title]
    res = call_llm_json_output(
        prompt_template,
        inputs={
            "content": content,
            "fact_strs": fact_strs
        },
        llm=llm,
        max_completion_tokens=1024,
    )
    res = json.loads(res)
    result = {
        "section_title": [],
        "fact": [],
        "result": [],
    }
    for k, v in res.items():
        k = int(k)
        result["fact"].append(facts[k])
        result["result"].append(v)
        result["section_title"].append(section_title)
    result = pd.DataFrame(result)
    return result

EXTRACT_PURPOSE_TEMPLATE = """
Your task is to extract key facts about Purpose of Research from the given clinical trial Informed Consent Form. You need to extract key facts that are only relevant to the Purpose of Research. 

The key facts should be related to number of participants, target conditions and treatments, or primary goal of the study. For each fact, you should be concise within a few sentences. 

You should extract no more than 5 facts and do not extract other unimportant facts such as general descriptions and other information unrelated to the purpose of research. Do not extract too specific details or requirements. Do not extract any information item that is not explicitly mentioned in the text. Return the extracted facts in a list.

Content:
{content}
"""

EXTRACT_RISK_TEMPLATE = """
Your task is to extract key facts about Possible Risks, Discomforts, and Inconveniences from the given clinical trial Informed Consent Form. You need to extract key facts that are only relevant to the Possible Risks, Discomforts, and Inconveniences of the trial. 

The key facts should be related to side effects or risks of discontinuing medications. For each fact, you should be concise within a few sentences. 

You should extract no more than 5 facts and do not extract other unimportant facts such as general descriptions and other information unrelated to the Possible Risks, Discomforts, and Inconveniences. Do not extract too specific details or requirements. Do not extract any information item that is not explicitly mentioned in the text. Return the extracted facts in a list.

Content:
{content}
"""

EXTRACT_DURATION_TEMPLATE = """
Your task is to extract key facts about Duration of Study Involvement from the given clinical trial Informed Consent Form. You need to extract key facts that are only relevant to the Duration of Study Involvement of the trial. 

Extract four facts: total study duration, active participation duration, follow-up and washout duration. For each fact, you should be concise within a few sentences. If any fact is not mentioned in the text, you can skip it. Do not extract other detailed settings or requirements.

Do not extract other unimportant facts such as general descriptions and other information unrelated to the Duration of Study Involvement. Do not extract too specific details, requirements and other uncommon settings. Do not extract any information item that is not explicitly mentioned in the text. Return the extracted facts in a list.

Content:
{content}
"""


EXTRACT_PROCEDURES_TEMPLATE = """
Content:
{content}

Your task is to list up to five key study procedures from the given clinical trial Informed Consent Form. 

For each procedure, you should just describe the name of the procedure and be concise within one sentence.
"""

def extract_facts(content, section_title, llm):
    """Extract key facts from the given content."""
    prompt_template = {
        "Purpose of Research": EXTRACT_PURPOSE_TEMPLATE,
        "Possible Risks, Discomforts, and Inconveniences": EXTRACT_RISK_TEMPLATE,
        "Duration of Study Involvement": EXTRACT_DURATION_TEMPLATE,
        "Procedures": EXTRACT_PROCEDURES_TEMPLATE
    }
    prompt = prompt_template[section_title]
    prompt += """

Your output should be a JSON object with the following format:
{{
    "facts": [ ... ] \\ list[str], the list of extracted facts
}}        
"""
    res = call_llm_json_output(
        prompt,
        inputs={
            "content": content
        },
        llm=llm,
        max_completion_tokens=1024,
    )
    res = json.loads(res)
    return res.get("facts", [])


if __name__ == "__main__":
    # example use case
    section_title = "Procedures"
    prompt = get_prompt_for_fact_extraction(section_title)
    print(prompt)

    # call llm
    # outputs = call_llm(
    #     prompt,
    #     inputs={
    #         "content": content
    #     },
    #     llm="gpt-4o-mini",
    #     temperature=0.01
    # )
    # print(outputs)