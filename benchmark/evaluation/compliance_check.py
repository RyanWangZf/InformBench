__all__ = [
    "validate_rules",
    "validate_icf_sections"
]


# defined by Ruba for icf generation experiment only
SECTION_NAME_TO_RULES = {
    "Purpose of Research": [
        {
            "rule_name": "1. Description of Clinical Investigation: 1a",
            "rule_description": "A clear and concise summary about the purpose of the research, what is being studied and what they hope to learn from conducting the study.",
        },
    ],
    "Duration of Study Involvement": [
        {
            "rule_name": "2. Duration of Study Involvement: 2a",
            "rule_description": "1h. The informed consent process must clearly describe the expected duration of the subject's participation in the clinical investigation (see 21 CFR 50.25(a)(1)), which includes their active participation as well as long-term follow-up, if appropriate.",
        }
    ],
    "Procedures": [
        {
            "rule_name": "1. Description of Clinical Investigation: 1c",
            "rule_description": "The description of the clinical investigation should identify tests or procedures required by the protocol; for example, drawing blood samples for a pharmacokinetic study or study surveys. ",
        },
        {
            "rule_name": "Procedures: a",
            "rule_description": "The procedures should be listed in logical order and organized by study phase or visit number. "
        },
        {
            "rule_name": "Procedures: b",
            "rule_description": "Include some details about study procedure that includes purpose of procedures."
        },
        {
            "rule_name": "Procedures: c",
            "rule_description": "1g. If there is follow-up period, prospective subjects must be informed of the procedures that will occur during such follow-up (21 CFR 50.25(a)(1))."
        }
    ],
    "Possible Risks, Discomforts, and Inconveniences": [
        {
            "rule_name": "2. Risks and Discomforts: 2a",
            "rule_description": "The informed consent process must describe the reasonably foreseeable risks or discomforts to the subject. This includes risks or discomforts of tests, interventions and procedures required by the protocol (including protocol-specified standard medical procedures, exams, and tests), with a particular focus on those that carry significant risk of morbidity or mortality."
        },
        {
            "rule_name": "2. Risks and Discomforts: 2d",
            "rule_description": "2d. Reasonably foreseeable discomforts related to procedures the subject may experience. For example, the consent form should disclose that the subject may be uncomfortable having to stay in one position or experience claustrophobia-like symptoms during an MRI."
        },
        {
            "rule_name": "2. Risks and Discomforts: 2e",
            "rule_description": "It is not necessary to describe all possible risks, especially if doing so could make the form overwhelming for subjects to read. Information on risks that are more likely to occur and those that are serious should be described so that prospective subjects can understand the nature of the risk"
        },
        {
            "rule_name": "2. Risks and Discomforts: 2g",
            "rule_description": "When appropriate, a statement must be included that a particular treatment or procedure may involve currently unforeseeable risks to the subject (or to the subject’s embryo or fetus, if the subject is or may become pregnant) (21 CFR 50.25(b)(1)."
        },
        {
            "rule_name": "2. Risks and Discomforts: 2h",
            "rule_description": "If the study is placebo-controlled study, include a statement that there may exist a risk that the disease/condition may go untreated and the subject’s condition may worsen"
        }
    ]
}

PROMPT_TEMPLATE_WO_RATIONALE = """Evaluate if the above content follows or violates the rules. Your assessment should be in two categories:

Y: follows the rule
N: violates the rule


Your outputs should be the JSON format:
```json
[
    {{
        "rule_id": "rule_0", \\ identifier of the given rule
        "prediction":  \\ in Y: follows the rule, N: violates the rule
    }},
    ... \\ more rules if more than one
]
```

Rules:
{rules}

Content:
{content}

Output:
"""

# Do your assessment step by step. For each rule, you need to first make reasoning to generate a rationale, then make a prediction based on the rationale.
# The rationale should be less than 20 tokens when the rule is violated (N)/not applicable (NA)/DNI; leave it blank if the rule is followed (Y).

import pdb
import json
import re
import logging
from benchmark.llm import call_llm_json_output

def _parse_list_of_json(data_str):
    """Parse a list of JSON objects from a string, e.g., [{...}, {...}, ...]
    """
    # Find all occurrences of `{...}` using regex
    matches = re.findall(r'\{.*?\}', data_str, re.DOTALL)

    # Parse each match separately
    parsed_items = []
    for match in matches:
        # Step 1: Replace single quotes around keys and values with double quotes
        json_compatible = re.sub(r"(?<![a-zA-Z0-9])'(.*?)'(?![a-zA-Z0-9])", r'"\1"', match)
        
        # Step 2: Replace single quotes inside values with a semicolon (;)
        json_compatible = re.sub(r"(?<!\\)'", ";", json_compatible)
        
        try:
            # Parse the JSON string
            parsed_item = json.loads(json_compatible)
            parsed_items.append(parsed_item)
        except json.JSONDecodeError as e:
            print(f"Failed to parse: {match}\nError: {e}")
            continue
    return parsed_items
    
def validate_rules(
    content: str,
    rules: list[dict],
    llm: str,
    ):
    """Assess if the given content follows or violates the given rules.

    Args:
        content (str): The content to be validated.
        rules (list[dict]): The rules to be validated. each rule has two keys: `rule_name` and `rule_description`.
        llm (str): The LLM model name.
    """
    rule_pred = {}
    try:
        rule_str =  [f"[rule:{i}] {r['rule_description']}" for i,r in enumerate(rules)]
        rule_str = "\n\n".join(rule_str)
        outputs = call_llm_json_output(
            PROMPT_TEMPLATE_WO_RATIONALE,
            inputs={
                "rules": rule_str,
                "content": content
            },
            llm=llm,
            temperature=0.01
        )

        # parse the outputs
        parsed_outputs = _parse_list_of_json(outputs)

        for output in parsed_outputs:
            # get the prediction for each rule
            rule_id = output.get("rule_id", "")
            try:
                idx = int(re.findall(r'\d+', rule_id)[0]) # find the number in the index
                rule_pred[idx] = {
                    "rationale": output.get("rationale", ""),
                    "prediction": output.get("prediction", "NA") # NA: not applicable
                }
            except ValueError:
                logging.error(f"Failed to get the rule index from the rule_id: {rule_id}")
                continue

    except Exception as e:
        logging.error(f"Failed to validate the content: {str(e)}")
        pass

    # get the prediction for each rule
    output_rules = []
    for i, rule in enumerate(rules):
        rule_pred_ = rule_pred.get(i, {"rationale": "", "prediction": "NA"})
        output_rules.append({
            "rule_name": rule["rule_name"],
            "rule_description": rule["rule_description"],
            "rationale": rule_pred_["rationale"],
            "prediction": rule_pred_["prediction"]
        })
    return output_rules

def get_section_name_to_rules():
    # STANDARD_SECTIONS = [
    # "Purpose of Research",
    # "Voluntary Participation",
    # "Duration of Study Involvement",
    # "Procedures",
    # "Participant Responsibilities",
    # "Withdrawal from Study",
    # "Possible Risks, Discomforts, and Inconveniences",
    # "Alternatives",
    # "Participant's Rights",
    # "Confidentiality"
    # ]
    # section_name_to_rules = {}
    # for section in STANFORD_ICF_TEMPLATE["sections"]:
    #     section_name = section["target_section"]
    #     validation_rules = section.get("validation_rules", [])
    #     if section_name in STANDARD_SECTIONS and len(validation_rules) > 0:
    #         # find the rules associated with each section
    #         section_name_to_rules[section_name] = validation_rules
    # return section_name_to_rules
    return SECTION_NAME_TO_RULES

def validate_icf_sections(section_title, content, llm):
    """Validate the given section content against the rules."""
    section_name_to_rules = get_section_name_to_rules()
    rules = section_name_to_rules.get(section_title, [])
    results = validate_rules(content, rules, llm)
    return results