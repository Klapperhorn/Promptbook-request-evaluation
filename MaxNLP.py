"""NLP and LLM work: clients (OpenAI, Nebula), promptbook coding, codebook -> JSON schema, evaluation against
manual coding (Krippendorff's alpha, disagreement sheets), text metrics, and causal-claim extraction.

Single maintained version: Git_Sync/_shared_modules (merged 2026-11-06 from MaxNLP.py, NLP_causal_mapping.py and the
text-metric functions of MR_Qualtrics_module). Credentials are read only when create_client() is called.
"""
import json
import re
from collections import Counter
from datetime import date

import jsonlines
import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------ LLM clients and promptbook coding


def create_client(client="Nebula", openAI_key=None):
    """OpenAI or Nebula client. Credentials come from openAI_key.py (gitignored) unless a module is passed."""
    from openai import OpenAI
    if openAI_key is None:
        import openAI_key

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


def eval_ai_vs_manual(merged_df, question_list, pred_prefix="ai", true_prefix="manual"):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as prfs, accuracy_score

    def one(topic):
        df = merged_df[[f"{pred_prefix}_{topic}", f"{true_prefix}_{topic}"]].dropna()
        y = df[f"{true_prefix}_{topic}"].astype(int)
        p = df[f"{pred_prefix}_{topic}"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        prec, rec, f1, _ = prfs(y, p, average="binary", zero_division=0)
        return pd.Series({"N": len(df),"N-pos": tp+fn, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                          "Precision": prec, "Recall": rec, "F1": f1, "F-beta": accuracy_score(y, p)})
    return pd.DataFrame({t: one(t) for t in question_list}).T


def kripp_alpha_all_variables(df, drop_non_coding_cols=None, decimals=3):
    import pandas as pd
    import numpy as np
    import krippendorff

    skip = set(drop_non_coding_cols or [])

    def assess(alpha):
        if pd.isna(alpha):
            return "not estimable"
        if alpha < 0:
            return "systematic disagreement"
        if alpha < 0.667:
            return "low reliability"
        if alpha < 0.800:
            return "tentative reliability"
        return "reliable"

    rows = []

    for col in [c for c in df.columns if c not in skip]:
        wide = df[col].unstack(0)
        vals = wide.stack().dropna()

        if vals.nunique() < 2:
            alpha = np.nan
        else:
            try:
                alpha = krippendorff.alpha(
                    reliability_data=wide.to_numpy().T,
                    level_of_measurement="nominal"
                )
            except Exception:
                alpha = np.nan

        rows.append((col, alpha, assess(alpha)))

    return (
        pd.DataFrame(rows, columns=["variable", "krippendorff_alpha", "assessment"])
        .set_index("variable")
        .sort_values("krippendorff_alpha", ascending=False, na_position="last")
        .round({"krippendorff_alpha": decimals})
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


def plot_text_duration_correlation(df, text_metrics, duration_col="Duration [Min]"):
    plt.figure(figsize=(12, 6))
    
    for col in [c for c in text_metrics.columns[-4:] if c.endswith('_chars')]:
        qid = col.split('_')[0]
        title = df[qid].iloc[0].split("?",1)[0] if qid in df.columns else qid

        sns.lmplot(
            x=duration_col,
            y=col,
            data=pd.concat([df[[duration_col]], text_metrics[[col]]], axis=1),
            scatter_kws={"s": 20, "alpha": 0.6},
            line_kws={"color": "red"},
            height=4,
            aspect=1.5
        ).set(title=title)


# ------------------------------------------------------------------ causal-claim extraction (formerly NLP_causal_mapping.py)


def make_causal_schema(
    cause_codes,
    actor_codes,
    edge_types=None,
    schema_name="causal_claims_response",
    max_claims=4,
    include_response_id=True,
    include_confidence=True,
    include_notes=True
):
    if edge_types is None:
        edge_types = ["CAUSES", "ENABLES", "WORSENS", "PREVENTS"]

    claim_properties = {
        "cause_node": {
            "type": "string",
            "enum": list(cause_codes)
        },
        "actor_attribution": {
            "type": "string",
            "enum": list(actor_codes)
        },
        "target": {
            "type": "string",
            "enum": list(cause_codes)
        },
        "edge_type": {
            "type": "string",
            "enum": list(edge_types)
        },
        "evidence": {
            "type": "string"
        }
    }

    claim_required = [
        "cause_node",
        "actor_attribution",
        "target",
        "edge_type",
        "evidence"
    ]

    if include_confidence:
        claim_properties["confidence"] = {
            "type": "string",
            "enum": ["high", "medium", "low"]
        }
        claim_required.append("confidence")

    schema_properties = {
        "has_causal_claim": {
            "type": "boolean"
        },
        "claims": {
            "type": "array",
            "maxItems": max_claims,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": claim_properties,
                "required": claim_required
            }
        }
    }

    schema_required = ["has_causal_claim", "claims"]

    if include_response_id:
        schema_properties["response_id"] = {
            "type": "string"
        }
        schema_required = ["response_id"] + schema_required

    if include_notes:
        schema_properties["notes"] = {
            "type": "string"
        }
        schema_required.append("notes")

    return {
        "name": schema_name,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": schema_properties,
            "required": schema_required
        }
    }


def get_content_from_response(response):
    choices = response["choices"] if isinstance(response, dict) else response.choices
    if not choices:
        return ""

    first = choices[0]

    if isinstance(first, list):
        first = first[0] if first else {}

    if isinstance(first, dict):
        message = first.get("message", {})
        content = message.get("content", "")
    else:
        message = getattr(first, "message", None)
        content = getattr(message, "content", "") if message is not None else ""

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        content = "\n".join(parts)

    return str(content).strip()


def extract_json_object(text):
    if pd.isna(text):
        return None

    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None

    return None


def code_one_response(
    text,
    response_id,
    client,
    model="FAST.gpt-oss:120b",
    SYSTEM_PROMPT="Assist the qualitative coding",
    SCHEMA=None,
    Temperature=0.0
):
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return {
            "response_id": str(response_id),
            "has_causal_claim": False,
            "claims": [],
            "notes": "empty response"
        }

    if SCHEMA is None:
        raise ValueError("SCHEMA must be provided.")

    props = SCHEMA["schema"]["properties"]["claims"]["items"]["properties"]
    cause_codes = props["cause_node"]["enum"]
    actor_codes = props["actor_attribution"]["enum"]
    target_codes = props["target"]["enum"]
    edge_types = props["edge_type"]["enum"]
    confidence_levels = props["confidence"]["enum"]

    response = client.chat.completions.create(
        model=model,
        temperature=Temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"response_id: {response_id}\n"
                    f"response_text: {text}\n\n"
                    "Return one JSON object with keys: response_id, has_causal_claim, claims, notes.\n"
                    f"Allowed cause_node/target codes: {cause_codes}\n"
                    f"Allowed actor_attribution codes: {actor_codes}\n"
                    f"Allowed edge_type values: {edge_types}\n"
                    f"Allowed confidence values: {confidence_levels}\n"
                    "Each claim must contain: cause_node, actor_attribution, target, edge_type, evidence, confidence.\n"
                    "Return JSON only. No markdown fences."
                )
            }
        ]
    )

    parsed = extract_json_object(get_content_from_response(response))
    if parsed is None:
        return {
            "response_id": str(response_id),
            "has_causal_claim": False,
            "claims": [],
            "notes": "parse_error"
        }

    parsed["response_id"] = str(response_id)
    parsed["has_causal_claim"] = bool(parsed.get("has_causal_claim", False))
    parsed["notes"] = str(parsed.get("notes", ""))
    parsed["claims"] = parsed.get("claims", [])

    valid_claims = []
    for c in parsed["claims"]:
        if (
            isinstance(c, dict)
            and c.get("cause_node") in cause_codes
            and c.get("actor_attribution") in actor_codes
            and c.get("target") in target_codes
            and c.get("edge_type") in edge_types
            and c.get("confidence") in confidence_levels
        ):
            valid_claims.append({
                "cause_node": c["cause_node"],
                "actor_attribution": c["actor_attribution"],
                "target": c["target"],
                "edge_type": c["edge_type"],
                "evidence": str(c.get("evidence", "")).strip(),
                "confidence": c["confidence"]
            })

    parsed["claims"] = valid_claims
    if parsed["claims"]:
        parsed["has_causal_claim"] = True

    return parsed


def plot_main_graph(G, min_weight=1, figsize=(12, 8), seed=42):
    H = nx.DiGraph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("weight", 0) >= min_weight
    )
    for n in H.nodes():
        H.nodes[n].update(G.nodes[n])

    pos = nx.spring_layout(H, seed=seed, k=1.2)

    node_sizes = [H.nodes[n].get("n_responses", 1) * 180 + 300 for n in H.nodes()]
    node_colors = [
        "#d95f02" if H.nodes[n].get("node_group") == "outcome" else "#1b9e77"
        for n in H.nodes()
    ]
    edge_widths = [H[u][v].get("weight", 1) * 0.7 for u, v in H.edges()]

    edge_color_map = {"CAUSES": "#377eb8", "ENABLES": "#984ea3", "WORSENS": "#e41a1c"}
    edge_colors = [edge_color_map.get(H[u][v].get("edge_type", ""), "#999999") for u, v in H.edges()]

    labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(
        H, pos, width=edge_widths, edge_color=edge_colors,
        arrows=True, arrowstyle="->", arrowsize=18, alpha=0.75,
        connectionstyle="arc3,rad=0.08"
    )
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=9)

    plt.title(f"Main causal graph (min edge weight = {min_weight})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_actor_graph(G, min_weight=1, figsize=(12, 8)):
    H = nx.DiGraph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("weight", 0) >= min_weight
    )
    for n in H.nodes():
        H.nodes[n].update(G.nodes[n])

    actors = [n for n in H.nodes() if H.nodes[n].get("node_group") == "actor"]
    causes = [n for n in H.nodes() if H.nodes[n].get("node_group") == "cause"]

    pos = {}
    for i, n in enumerate(sorted(actors)):
        pos[n] = (0, -i)
    for i, n in enumerate(sorted(causes)):
        pos[n] = (3, -i)

    node_sizes = [H.nodes[n].get("n_responses", 1) * 180 + 300 for n in H.nodes()]
    node_colors = ["#7570b3" if n in actors else "#1b9e77" for n in H.nodes()]
    edge_widths = [H[u][v].get("weight", 1) * 0.7 for u, v in H.edges()]
    labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(
        H, pos, width=edge_widths, edge_color="#666666",
        arrows=True, arrowstyle="->", arrowsize=18, alpha=0.7
    )
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=9)

    plt.title(f"Actor attribution graph (min edge weight = {min_weight})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def flatten_claim_sets_for_alpha(df):
    import pandas as pd

    x = df.copy()
    x["claims"] = x["claims"].apply(lambda v: v if isinstance(v, list) else [])

    def extract_set(lst, key):
        return sorted({d.get(key) for d in lst if isinstance(d, dict) and pd.notna(d.get(key))})

    x["cause_set"] = x["claims"].apply(lambda lst: extract_set(lst, "cause_node"))
    x["actor_set"] = x["claims"].apply(lambda lst: extract_set(lst, "actor_attribution"))
    x["target_set"] = x["claims"].apply(lambda lst: extract_set(lst, "target"))
    x["edge_set"] = x["claims"].apply(lambda lst: extract_set(lst, "edge_type"))

    out = x[["has_causal_claim"]].copy()

    for prefix, col in [
        ("cause", "cause_set"),
        ("actor", "actor_set"),
        ("target", "target_set"),
        ("edge", "edge_set"),
    ]:
        exploded = x[[col]].explode(col)
        dummies = pd.get_dummies(exploded[col], prefix=prefix).groupby(level=[0, 1]).max()
        out = out.join(dummies, how="left")

    return out.fillna(0).astype(int)
