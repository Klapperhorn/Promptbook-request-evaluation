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
        y = df[f"manual_{topic}"].astype(int)
        p = df[f"ai_{topic}"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        prec, rec, f1, _ = prfs(y, p, average="binary", zero_division=0)
        return pd.Series({"N": len(df),"N-pos": tp+fn, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                          "Precision": prec, "Recall": rec, "F1": f1, "F-beta": accuracy_score(y, p)})
    return pd.DataFrame({t: one(t) for t in question_list}).T


def make_one_text(texts,detailed=True):
    texts=texts.fillna("")
    meaning=texts["QID5"]
    causes=texts["QID8"]
    response=texts["QID10"]
    consequence=texts["QID15"]
    moral=texts["QID19"]
    if detailed==True:
        
        text_long=f"The energy crisis: {meaning};\n It was caused by: {causes} ;\n policy makers responded to the crisis: {response};\n this resulted in: {consequence};\n This should therefore be done: {moral}"
        return text_long
    else:
        text=f"{meaning}. {causes}. {response}. {consequence}. {moral}"
    
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
              temperature=0.0, force_json_object=False, batch_size=50, out_dir="annotations_tmp",run_x=0,codebook_name="causes"):
    
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
    file = f"{out_dir}/{date_str} run_{run_x}_{codebook_name}_batch_{batch_row // batch_size:03d}.jsonl"
    
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

def kripp_alpha_all_variables(df, drop_non_coding_cols=None, decimals=3):
    import pandas as pd
    import numpy as np
    import krippendorff

    skip = set(drop_non_coding_cols or [])

    def assess(a):
        if pd.isna(a):
            return "not estimable"
        elif a < 0:
            return "systematic disagreement"
        elif a < 0.667:
            return "low reliability"
        elif a < 0.800:
            return "tentative reliability"
        else:
            return "reliable"

    rows = []
    n_runs_out = None

    for col in df.columns.difference(skip):
        w = df[col].unstack(0)
        n_items, n_runs = w.shape
        n_runs_out = n_runs

        vals = pd.unique(w.stack().dropna())
        if len(vals) < 2:
            a = np.nan
        else:
            try:
                a = krippendorff.alpha(
                    reliability_data=w.to_numpy().T,
                    level_of_measurement="nominal"
                )
            except Exception:
                a = np.nan

        if pd.api.types.is_numeric_dtype(w):
            p = int((w.mean(axis=1) > 0.5).sum())
            pr = p / n_items if n_items else np.nan
        else:
            p, pr = np.nan, np.nan

        rows.append([col, a, assess(a), p, pr])

    print(f"n_runs: {n_runs_out}")

    return (
        pd.DataFrame(
            rows,
            columns=[
                "variable", "krippendorff_alpha", "assessment",
                "positive_count", "positive_rate"
            ]
        )
        .set_index("variable")
        .sort_values("krippendorff_alpha", ascending=False, na_position="last")
        .round({"krippendorff_alpha": decimals, "positive_rate": decimals})
    )


def write_disagreement_excel(annotation_df, merged_df, codes, path):
    import xlsxwriter
    """
    annotation_df: rows = items, columns = codes + other cols.
    merged_df: rows = items, with ai_<code>, manual_<code>.
    codes: list of code names (matching code rows after transpose).
    """
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # Transpose for your preferred layout: rows = codes, cols = items
        annotation_T = annotation_df.T.where(annotation_df.T.notna(), None)
        annotation_T.to_excel(
            writer,
            sheet_name="Disagreements",
            freeze_panes=(1, 1)
        )

        wb = writer.book
        ws = writer.sheets["Disagreements"]

        fp_fmt = wb.add_format({"bg_color": "#FF9999"})  # FP = light red
        fn_fmt = wb.add_format({"bg_color": "#99CCFF"})  # FN = light blue

        # annotation_T.index: codes + other rows (text, notes, ...)
        # annotation_T.columns: items (must match merged_df.index)
        for row_excel, code in enumerate(annotation_T.index, start=1):
            if code not in codes:
                continue  # only color true code rows

            ai_col = f"ai_{code}"
            manual_col = f"manual_{code}"
            if ai_col not in merged_df.columns or manual_col not in merged_df.columns:
                continue

            for col_excel, item in enumerate(annotation_T.columns, start=1):
                if item not in merged_df.index:
                    continue

                # value as stored in Annotation (not from merged_df)
                cell_val = annotation_T.loc[code, item]

                ai_val = merged_df.loc[item, ai_col]
                manual_val = merged_df.loc[item, manual_col]

                if (ai_val == 1) and (manual_val == 0):       # FP
                    ws.write(row_excel, col_excel, cell_val, fp_fmt)
                elif (ai_val == 0) and (manual_val == 1):     # FN
                    ws.write(row_excel, col_excel, cell_val, fn_fmt)
                else:
                    ws.write(row_excel, col_excel, cell_val)

