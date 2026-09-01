import pandas as pnd
import xml.etree.ElementTree as ET
import re

import sys
import traceback
import json

def return_elem(p_json, field):
    if field in p_json:
        return p_json[field]
    else:
        return None

in_xml="notices_json_ld_nl.xml"

out="proche_notices_json_ld_fr_nl_en.xlsx"

tree=ET.parse(in_xml)
i=0
target=pnd.DataFrame()

for person in tree.iter('person'):
    print(i)
    print(person)
    match1 = re.compile('\r|\n|\r\n')
    display_name=person.find("display_name").text
    text_entry=person.find("text_entry").text
    text_entry=match1.sub("", text_entry)
    author_notice=person.find("author_notice").text
    entered_date=person.find("entered_date").text 
    description_en=person.find("description_en").text   
    description_nl=person.find("description_nl").text       
    json_ld=person.find("json_ld").text    
    try:
        tmp_json=json.loads(json_ld)
        birth_place=return_elem(tmp_json,"birthPlace")
        death_place=return_elem(tmp_json,"deathPlace")
        birth_date=return_elem(tmp_json,"birthDate")
        death_date=return_elem(tmp_json,"deathDate")
        nationality=return_elem(tmp_json,"nationality")
        work_places=return_elem(tmp_json,"workLocation")
        affiliation=return_elem(tmp_json,"worksFor")
        description_fr=return_elem(tmp_json,"description")
        print(display_name)
        print(work_places)
        
        print(tmp_json)
        row=pnd.DataFrame([{"display_name":display_name, "birth_place":birth_place, "death_place": death_place, "birth_date":birth_date, "death_date":death_date , "nationality": nationality, "work_places":";".join(work_places), "description_fr":description_fr, "description_nl":description_nl, "description_en":description_en, "original_text":text_entry, "author_notice":author_notice, "entered_date":entered_date}])
        target = pnd.concat([target, row], ignore_index=True)
    except Exception:
        print(traceback.format_exc())
    i=i+1
print(target)
target.to_excel(out)
