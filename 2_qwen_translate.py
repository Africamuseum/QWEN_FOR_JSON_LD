import xml.etree.ElementTree as ET
import re

import sys
import traceback
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import copy


in_xml="notices_json_ld.xml"
out_xml="notices_json_ld_en_nl.xml"
model_path="local_models/qwen_7b_vl"

def translate(model, processor, p_text, p_lang_1, p_lang_2):
    messages = [{"role": "user", "content": "Translate the following text from "+p_lang_1+" to "+p_lang_2+" :"+p_text}
            ]
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

tree=ET.parse(in_xml)
root2 = ET.Element("root")


i=0
for person in tree.iter('person'):
    print(i)
    print(person)    
    json_ld=person.find("json_ld").text
    
    try:
        tmp_json=json.loads(json_ld)
        #print(tmp_json)
        print(len(tmp_json))
        if "description" in tmp_json:
            description=tmp_json["description"]
            print(description)
            resp=translate(model, processor, description, "French", "Dutch")
            if isinstance(resp,list):
                resp="".join(resp)
            print(resp)
            person2=copy.deepcopy(person)
            ET.SubElement(person2, "description_nl").text = resp
            root2.append(person2)
            
    except Exception:
        print(traceback.format_exc())
        
    
    i=i+1
    
tree2 = ET.ElementTree(root2)
print(tree2)
with open(out_xml, 'wb') as f:
    tree2.write(f)       