The traceback is coming out of the HF data‐collator’s call to `tokenizer.pad(...)`, and if you look closely it’s choking on your “query” field (which is a Python string) rather than on the token IDs:

  • In your `map(tokenize_prompt)` step you returned  
      `{ "input_ids": [ints…],  "query": "some prompt text" }`  
    
  • Then, since you passed `data_collator=None`, the trainer falls back to  
    `DataCollatorWithPadding(tokenizer)`.  That collator does roughly:  
      `batch = tokenizer.pad(features, return_tensors="pt")`  
    where each `feature` is a dict with both `input_ids` (ints) _and_ `query` (a str).  

  • Under the hood `tokenizer.pad` tries to turn _every_ value in `features` into a tensor.  When it hits your list of _strings_ it does  
      `torch.tensor(value)`  
    and blows up with  
      `ValueError: too many dimensions 'str'`  
    because you can’t make a 1D Torch tensor out of Python strings.  

In short, the collator is trying to pad/string-tensorize your raw prompt text, which it can’t do.  You need to either:

  1. Remove `"query"` from what gets fed into the collator (e.g. only return `input_ids` in your dataset and stash the raw strings elsewhere),  
  2. Or supply a custom `data_collator` that only pads the numeric inputs (`input_ids`, `attention_mask`, etc.) and leaves `"query"` alone (e.g. collect it into a Python list).

Once your collator only sees lists of integers (the token IDs), you’ll stop getting the “too many dimensions ’str’” error.
