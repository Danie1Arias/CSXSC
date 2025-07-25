from datasets import load_dataset

ds_guiacat = load_dataset("projecte-aina/GuiaCat")
ds_cassa = load_dataset("projecte-aina/CaSSA-catalan-structured-sentiment-analysis")

ds = ds_guiacat.remove_columns([col for col in ds_guiacat["train"].column_names if col not in ["text", "label"]])

def classify_guiacat(ds):
    for split in ds:
        for i in range(len(ds[split])):
            label = ds[split][i]["label"]
            if label in ["molt bo", "bo"]:
                ds[split][i]["label"] = "positive"
            elif label == "regular":
                ds[split][i]["label"] = "neutral"
            elif label in ["dolent", "molt dolent"]:
                ds[split][i]["label"] = "negative"
    return ds
    

ds = classify_guiacat(ds)
print(ds["train"][0])

