import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def llada_process_docs(dataset):
    def _process_doc(doc):
        ctx = doc.get("ctx_a", "") + " " + doc.get("ctx_b", "").capitalize()
        if "activity_label" in doc:
            ctx = f"{doc['activity_label']}: {ctx}"
        ctx = preprocess(ctx)

        endings = doc.get("endings") or doc.get("choices") or []
        endings = [preprocess(e) for e in endings]
        ending_map = dict(zip(["A", "B", "C", "D"], endings + [""] * (4 - len(endings))))
        label = int(doc.get("label", doc.get("gold", 0)))

        # Return both A-D and choices list
        return dict(ctx=ctx, choices=endings, **ending_map, label=label)

    return dataset.map(_process_doc)
