import pandas as pd
import numpy as np
import re
import openAI_key

def create_client(client="Nebula", openAI_key=openAI_key):
    from openai import OpenAI

    if client!="Nebula":
        client = OpenAI(
          api_key=openAI_key.key,
          organization=openAI_key.organization,
          project=openAI_key.project,
        )
        print("OpenAI client loaded")
        
    if client=="Nebula":
        client= OpenAI(
        base_url=openAI_key.NEBULA_BASE_URL, 
        api_key=openAI_key.NEBULA_API_KEY
        )
        
        print("Nebula client loaded")
        
    return client

def get_nebula_models(client):
    models = []
    for model in client.models.list().data:
        models.append(model.id)
    return models

def extract_text_metrics(text):
    if isinstance(text, str):
        chars = len(text)
        words = len(text.split())
        sents = len(re.findall(r'[.!?]', text))
        return chars, words, sents
    return 0, 0, 0

def compute_text_metrics(df, text_cols, duration_col="Duration [Min]"):
    metrics = ['chars', 'words', 'sents']
    
    out = pd.DataFrame(index=df.index)
    print(out)
    # Step 1: Extract row-level metrics
    for col in text_cols:
        out[[f"{col}_{m}" for m in metrics]] = df[col].apply(extract_text_metrics).tolist()
        out[f"{col}_chars_per_min"] = out[f"{col}_chars"] / df[duration_col].replace(0, np.nan)

    return out

def summarize_metrics(metrics_df, duration_series):
    summary = {}
    for col in metrics_df.columns:
        summary[col] = {
            'min': metrics_df[col].min(),
            'mean': metrics_df[col].mean(),
            'median': metrics_df[col].median(),
            'corr_with_duration': metrics_df[col].corr(duration_series)
        }
    return pd.DataFrame(summary).T

def eval_ai_vs_manual(merged_df, question_list):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as prfs, accuracy_score

    def one(topic):
        df = merged_df[[f"ai_{topic}", f"manual_{topic}"]].dropna()
        y = df[f"ai_{topic}"].astype(int)
        p = df[f"manual_{topic}"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        prec, rec, f1, _ = prfs(y, p, average="binary", zero_division=0)
        return pd.Series({"N": len(df), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                          "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": accuracy_score(y, p)})
    return pd.DataFrame({t: one(t) for t in question_list}).T


def make_one_text(texts):
    texts=texts.fillna("")
    meaning=texts["QID5"]
    causes=texts["QID8"]
    response=texts["QID10"]
    consequence=texts["QID15"]
    moral=texts["QID19"]
    text_long=f"The energy crisis was evident: {meaning};\n The crisis was caused by: {causes} ;\n policy makers responded to the crisis: {response};\n which finally resulted in: {consequence};\n This should therefore be done: {moral}"
    text=f"{meaning};\n caused by: {causes} ;\n crisis resonses: {response};\n consequences: {consequence};\n implications: {moral}"
    
    return  text

def make_json_schema(codebook="", codes=[]):
    return {
        "name": f"{codebook}_likelihoods",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {c: {"type": "number", "minimum": 0.0, "maximum": 1.0} for c in codes},
            "required": list(codes),
        },
    }


def openAI_code_text(text,model="gpt-4.1-mini",SCHEMA={},PROMPTBOOK_INSTRUCTIONS="") -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,  # slight temp >0 helps calibration [web:96]
        messages=[
            {"role": "system", "content": PROMPTBOOK_INSTRUCTIONS},  # use revised instructions above
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA["name"],
                "strict": True,
                "schema": SCHEMA["schema"]
            }
        },
    )
    return json.loads(resp.choices[0].message.content)

### not used atm.
def nebula_code_text(model="deepseek-r1:8b", SCHEMA={}, PROMPTBOOK_INSTRUCTIONS="", text="", configs=None):
    prompt_parameters = {
        "model": model,
    #    "temperature":0.1,  # slight temp >0 helps calibration [web:96]
        "messages": [
            { "role": "system", "content": PROMPTBOOK_INSTRUCTIONS },
            { "role": "user", "content": text }
        ],
        "response_format":{
                    "type": "json_schema",
                    "json_schema": {
                        "name": SCHEMA["name"],
                        "strict": True,
                        "schema": SCHEMA["schema"]
                    }
        }
    }
    if configs:
        prompt_parameters.update(configs)

    response = NEBULA.chat.completions.create(**prompt_parameters)

    return json.loads(response.choices[0].message.content)
### not used atm.    


### uses Batches --> used atm.

import json
import jsonlines
from datetime import date

def code_text(text, orig_index, batch_row, client, model, code_list, PROMPTBOOK_INSTRUCTIONS,
              temperature=0.0, force_json_object=False, batch_size=50, out_dir="annotations_tmp",run_x=0):
    
    SCHEMA = make_json_schema("code", code_list)
        
    date_str = date.today().isoformat()
    
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": PROMPTBOOK_INSTRUCTIONS},
            {"role": "user", "content": text},
        ],
        response_format=(
            {"type": "json_object"} if force_json_object else
            {"type": "json_schema",
             "json_schema": {"name": SCHEMA["name"], "strict": True, "schema": SCHEMA["schema"]}}
        ),
    )

    # write batch files
    parsed = json.loads(resp.choices[0].message.content)
    file = f"{out_dir}/{date_str} run_{run_x}_causes_batch_{batch_row // batch_size:03d}.jsonl"
    
    with jsonlines.open(file, mode="a") as writer:
        writer.write({"orig_index": orig_index, "annotation": parsed})

    return parsed

def markdown_to_json(md_text: str) -> str:
    import re
    import json
    
    codes = []
    sections = re.split(r'(?=^##\s*Code:\s*.+?$)', md_text, flags=re.MULTILINE)
    
    for section in sections:
        section = section.strip()
        if not re.match(r'##\s*Code:', section):
            continue
        
        label_match = re.search(r'##\s*Code:\s*(.+?)(?=\n)', section, re.MULTILINE)
        label = label_match.group(1).strip() if label_match else ''
        
        content = re.sub(r'^##\s*Code:\s*.+\n?', '', section, flags=re.MULTILINE).strip()
        code = {'label': label}
        
        # Description - collapse whitespace
        desc_match = re.search(r'Description:\s*(.*?)(?=^\s*Inclusion criteria:|^\s*Exclusion criteria:|^\s*Included examples:|\Z)', content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        code['description'] = re.sub(r'[\n\r\s]+', ' ', desc_match.group(1).strip()) if desc_match else ''
        
        # Lists - simple bullet extraction
        for field, pattern in [
            ('inclusion_criteria', r'Inclusion criteria:\s*(.+?)(?=^Exclusion criteria:)'),
            ('exclusion_criteria', r'Exclusion criteria:\s*(.+?)(?=^Included examples:|\Z)'),
        ]:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            items = re.findall(r'-\s*([^\n]+)', match.group(1) if match else '', re.MULTILINE)
            code[field] = [item.strip() for item in items if item.strip()]
        
        # Examples - rest of content
        examp_match = re.search(r'Included examples:\s*(.+)', content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        examp_items = re.findall(r'-\s*([^\n]+)', examp_match.group(1) if examp_match else '', re.MULTILINE)
        code['included_examples'] = [item.strip() for item in examp_items if item.strip()]
        
        codes.append(code)
    
    return codes



