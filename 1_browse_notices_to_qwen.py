import xml.etree.ElementTree as ET
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import sys
import traceback
import json




in_xml="notices.xml"
out_xml="notices_json_ld.xml"
model_path="local_models/qwen_7b_vl"

def clean_text(text):
    tmp=text.split(".")
    unq=list(set(tmp))
    return ".".join(unq)
    
    
def summarize(model, processor, messages, p_doc, text_entry):
    returned=p_doc
    try:
        tmp=json.loads(p_doc)
        if "description" in tmp:
            old_description=tmp["description"]
            messages = [
            {"role": "user", "content": "Summarize this French text. Keep the result in French language. Try to use less than 100 words :"+text_entry}
            ]
            resp=go_model(model, processor, messages) 
            print("--------")
            print(resp)
            if resp is not None:
                if isinstance(resp, list):
                    resp="".join(resp)
                if resp is not None:    
                    count = len(re.findall(r'\w+', resp))
                    if count < 200:
                        tmp["description"]=resp
                    elif old_description>200:
                        tmp["description"]=clean_text(old_description)
                returned=json.dumps(tmp)
    except Exception:
        print(traceback.format_exc())
    finally:
        return returned
    

def go_model(model, processor, messages):
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
                    text=[text_prompt],
                    return_tensors="pt",
                )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=8092)
    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
    output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
    print("result")
    print(output_text)
    return output_text


tree=ET.parse(in_xml)
print(tree)

regex = re.compile(r"(\u2009|\u202f)", re.UNICODE)
i=0




quant_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16
)
     
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto",
    quantization_config=quant_config,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path)
question="""Summarize the following text and convert it into JSON-LD, using schema.org for persons. Keep the text in French. Here is an example of output structure: {
  "@context": "https://schema.org",
  "@type": "Person",
  "@id": "",
  "name": "",
  "birthDate": "",
  "deathDate": "",
  "birthPlace": "",
  "deathPlace": "",
  "nationality": "",
  "affiliation": "",
  "workLocation": [""],
  "description": "",
  "creator": {
    "@type": "Person",
    "name": ""
  },
  "dateCreated": "",
  "citation": [
    ""
  ]
}
. The text is:"""


    
root2 = ET.Element("root")
i=0
for person in tree.iter('person'):
    
    print(person)
    display_name=person.find("display_name").text
    text_entry=person.find("text_entry").text
    author_notice=person.find("author_notice").text
    entered_date=person.find("entered_date").text
    text_entry= regex.sub(r" ",text_entry) #regex.sub(" ",text_entry)
    print(display_name)
    print(text_entry)
    messages = [
    {"role": "user", "content": question+text_entry}
    ]
    resp=go_model(model, processor, messages)
    
    doc = ET.SubElement(root2, "person")
    ET.SubElement(doc, "display_name").text = display_name
    ET.SubElement(doc, "text_entry").text = text_entry
    ET.SubElement(doc, "author_notice").text = author_notice
    ET.SubElement(doc, "entered_date").text = entered_date
    if isinstance(resp, list):
        resp="".join(resp)
    resp=summarize(model, processor, messages, resp, text_entry)
    ET.SubElement(doc, "json_ld").text = resp
 
    
    i=i+1
    """
    tmp=text_entry.split(" ")
    for x in tmp:
        uni= x.encode("unicode_escape")
        print(x+"\t"+str(uni))
    """

    
tree2 = ET.ElementTree(root2)
print(tree2)
with open(out_xml, 'wb') as f:
    tree2.write(f)   
