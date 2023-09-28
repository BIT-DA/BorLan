import torch
import os
import sys
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model, XLNetTokenizer, XLNetModel, DebertaTokenizer, DebertaModel, MT5Model, GPT2Model, GPT2Tokenizer
from classnames import *
from imagenet_templates import IMAGENET_TEMPLATES, extra_templates
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# -----------------------------------------------------------------------
# Setting up here.
choice = 'BertL' # Choose language model
classnames = CLASSES_Aircraft # Set classnames
save_name = 'aircraft_BertL.pt' # filename for text embedding

object_token_avg = True # use [CLS] token or word token
use_prefix = True # add prefix "This is"
# -----------------------------------------------------------------------

model_dict = {'example':['model','tokenizer','model-type','tokenizer-type'],
              'Bert':['BertModel','BertTokenizer','prajjwal1/bert-small','prajjwal1/bert-small'],
              'BertB':['BertModel','BertTokenizer','bert-base-uncased','bert-base-uncased'],
              'BertL':['BertModel','BertTokenizer','bert-large-uncased','bert-large-uncased'],
              'Deberta':['DebertaModel','DebertaTokenizer','microsoft/deberta-large','microsoft/deberta-large'],
              'XLNet':['XLNetModel','XLNetTokenizer','xlnet-large-cased','xlnet-large-cased'],
              'T5':['T5Model','T5Tokenizer','t5-large','t5-large'],
              'mT5':['MT5Model','T5Tokenizer','google/mt5-large','google/mt5-large'],
              'T5-3b':['T5Model','T5Tokenizer','t5-3b','t5-3b'],
              'gpt2':['GPT2Model','GPT2Tokenizer','gpt2','gpt2'],} #GPT suggest using text_embedding_gpt.py instead


tokenizer = eval(model_dict[choice][1]).from_pretrained(model_dict[choice][3])
model = eval(model_dict[choice][0]).from_pretrained(model_dict[choice][2]).cuda()
model.eval()

templates = IMAGENET_TEMPLATES
# templates = IMAGENET_TEMPLATES+extra_templates

# remember the token locations corresponding to classname
prefix_len = []
for temp in templates:
    if use_prefix:
        temp = 'This is ' + temp if temp.startswith('a') or temp.startswith('the') else temp 
    temp_prefix = temp.split('{}')[0]
    # print(temp_prefix)
    prefix_token = tokenizer.tokenize(temp_prefix)
    prefix_len.append(len(prefix_token))

class_len = []
for c in classnames:
    c = c.replace('_', ' ')
    class_token = tokenizer.tokenize(c)
    class_len.append(len(class_token))
# print(prefix_len, class_len)

def article(name):
  return 'an' if name[0] in 'aeiouAEIOU' else 'a'

with torch.no_grad():
    all_text_features = []
    for i, temp in enumerate(templates):
        if use_prefix:
            temp = 'This is ' + temp if temp.startswith('a') or temp.startswith('the') else temp 
        prompts = [temp.format(c.replace("_", " "),article=article(c)) for c in classnames]
        tokens = tokenizer(prompts, return_tensors='pt', padding=True)
        # print(tokens['input_ids'])
        batch = {'input_ids': tokens['input_ids'].cuda(), 'attention_mask': tokens['attention_mask'].cuda()}
        # print(batch)
        # print(tokenizer.decode(tokens['input_ids'][0]))
        if choice in ['Bert','BertB','BertL','Deberta','XLNet','gpt2']:
            output = model(input_ids=tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda(), return_dict = True)
        else:
            output = model.encoder(**batch, return_dict = True)
        last_hidden_states = output.last_hidden_state
        # print(last_hidden_states.size())
        if object_token_avg:
            text_features = []
            for j, out in enumerate(last_hidden_states):
                # print(out.size(), prefix_len[i], class_len[j])
                text_feature = out[prefix_len[i]+1:prefix_len[i]+class_len[j]+1]
                # print(j,text_feature.size())
                text_feature = text_feature.mean(dim=0)
                text_features.append(text_feature)
            text_features = torch.stack(tuple(text_features), dim=0)
        else:
            print('use cls token')
            text_features = last_hidden_states[:,0,:].squeeze()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(text_features.size())
        # print("debug size", text_features.size())
        all_text_features.append(text_features)
        # break
    all_text_features = torch.stack(tuple(all_text_features),dim=1)
    print(all_text_features.size())
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

torch.save(all_text_features.cpu(), save_name)
print('text embedding '+save_name + ' is saved.')
