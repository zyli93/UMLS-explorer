# Setup Pubmed Parser

## Download MEDLINE and PubMed files

Check out here: https://github.com/titipata/pubmed_parser/wiki

## Install packages

Just simply run (assuming in interactive docker container with `root` privilege
```bash
# bash setup_env.sh
```

## Load MEDLINE
The medline files are stored in this location: `/local2/zyli/umls/MEDLINE/`.
Run python code as the following:
```python
import pubmed_parser as pp

dicts_out = pp.parse_medline_xml("./MEDLINE/pubmed19n0001.xml.gz", 
                                 year_info_only=False,
                                 nlm_category=False,
                                 author_list=False,
                                 reference_list=False)
# the dicts_out is a list of dictionaries, 30000 entries in total

dict0 = dicts_out[0]  # take the first element

print(dict0.keys())
# output: dict_keys(['publication_types', 'medline_ta', 'journal', 'issn_linking', 'authors', ...
# only the `abstract` domain is a text domain.

# only a part of entries have `abstract`
tmp = [1 if len(x["abstract"])==0 for x in dicts_out]
sum(tmp)
# output: 14623
```



