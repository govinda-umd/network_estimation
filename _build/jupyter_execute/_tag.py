#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nbformat as nbf
from glob import glob
from tqdm.notebook import tqdm

# Collect a list of all notebooks in the content folder
nb_name = 'jan22'
notebooks = list(filter(
    lambda x: nb_name in x,
    glob("./nb/**/*.ipynb", recursive=True)
))

# exclude = ['./_tag.ipynb', './_reset_cell_id.ipynb']
# notebooks = [item for item in notebooks if item not in exclude]

# Text to look for in adding tags
text_search_dict = {
    "# HIDDEN": "remove-cell",  # Remove the whole cell
    "# NO CODE": "remove-input",  # Remove only the input
    "# HIDE CODE": "hide-input"  # Hide the input w/ a button to show
}

# Search through each notebook and look for the text, add a tag if necessary
pbar = tqdm(sorted(notebooks), leave=False)
for ipath in pbar:
    pbar.set_description(f"tagging {ipath}")
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        cell_tags = cell.get('metadata', {}).get('tags', [])
        for key, val in text_search_dict.items():
            if key in cell['source']:
                if val not in cell_tags:
                    cell_tags.append(val)
        if len(cell_tags) > 0:
            cell['metadata']['tags'] = cell_tags

    nbf.write(ntbk, ipath)

