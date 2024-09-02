import os, logging, sys, json
import re, torch, shutil, copy
from datasets import load_dataset
from multiprocess import set_start_method
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from datasets.utils.logging import disable_progress_bar
from rclone_python import rclone

'''
---------------
Creating embeddings for any general dataset/json file in the following format 
{'wos_id': {'abstract', 'item', ... (other metadata)}}
Multiprocessing is not written in this code because we only had one GPU (can be modified in you have multiple)
'''

class embedder:
    def __init__(self, file_path, temp_path, result_path,
                 cache, log_file, corpus_path = None, emb_corpus = None):
        #Path to the dir with an initial file (the raw json dict)
        self.file_path = file_path
        #Path where temp files are stored (they will be deleted by the time this function stop running)
        self.temp_path = temp_path
        #Corpus path 
        self.corpus_path = corpus_path
        #Path to the resulting file 
        self.result_path = result_path
        #Embedded files 
        self.emb_corpus = emb_corpus
        #Path to cache storage (should be in a place with a lot of available space) -- it will be cleared at the end of the cycle
        self.cache = cache
        #Log file 
        self.log = log_file
        
        
def read_json_file(file_path):
    f_dict = {}
    """
    Read a JSON file with dictionaries separated by commas.
    
    This only needs to be used after embedding

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries parsed from the file.
    """
    with open(file_path, 'r') as f:
        json_dicts = [json.loads(line) for line in f]
    for rec in json_dicts:
        f_dict[rec['wos_id']] = copy.deepcopy(rec['embeddings'])
    return f_dict


def transform_json(path, drop_keys, drop_abstractless = False):
    '''HF Datasets only reads jsons of the following format 
    {'id1', 'text1', 'other_metadata}
    {'id2', 'text2', 'other_metadata}
    This is not compatible with the format wos records are stored in
    {'r_id1': {metadata}}
    This function reforms jsons into the needed format and removes unnecessary fields (For memory effeciency)'''
    #filtering out unnecessary metadata
    vital_keys = ['wos_id', 'item', 'abstract']
    final_list = []
    with open(path, "r") as gg:
        data = json.load(gg)
    for key in data.keys():
        data[key]['wos_id'] = copy.deepcopy(key)
        if drop_keys:
            subset_dict = {k: data[key][k] for k in vital_keys}
        else:
            subset_dict = data[key]
        if drop_abstractless and subset_dict['abstract'] == "":
            continue
        #These fields can confused because dataset does not like dicts within dicts 
        #Converting them into string allows to preserve the data
        for wk in ['citations', 'yearly_cits', 'yearly_cit', 'addresses']:
            if wk in subset_dict.keys():
                subset_dict[wk] = copy.deepcopy(str(subset_dict[wk]))
        
        final_list.append(subset_dict)
    with open(path, "w") as dson:
        json.dump(final_list, dson)
    del final_list
    del data 
    
def init_models():
    #Initializing tokenizer, model, and adapter
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    adapter_name = model.load_adapter("allenai/specter2", source="hf", set_active=True)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    return tokenizer, model, adapter_name



def embed(self, tokenizer, model, adapter_name, device, out_format = "json",
          drop_keys = True, drop_abstractless = False):
    '''Creates embeddings in a new column
    Input -- a json file with article records (IT SHOULD HAVE ITEM (TITLE) AND ABSTRACT)
    Output -- records with embeddings
    drop_keys -- only set this to True if working with very large files. 
    If true, the output will only have these five fields -- wos_id, abstract, item, full_text, embeddings'''
    def concatenate_text(examples, tokenizer = tokenizer):
    #The model accepts concatenated title + abstracts strings 
    #We need to get all files to this format
        return {
            "full_text": examples["item"]
            + tokenizer.sep_token
            + examples["abstract"]
        }
    def pre_process(dsf):
        #Applies the function above to the entire dataset
        return dsf.map(concatenate_text)
    #The main embedding function
    def get_embeddings(text_list, device = device, tokenizer = tokenizer):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt",
            return_token_type_ids=False, max_length=512)
        model.to(device)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return model_output.last_hidden_state[:, 0, :]
    num_cores = torch.cuda.device_count()
    print(f"The number of cores is {num_cores}")
    file_names = [file['Path'] for file in rclone.ls(self.file_path)]
    for file_name in file_names:
        base_name = os.path.basename(file_name).replace(".json", '').replace(".csv", '')
        format = re.search("[a-z]{3,4}$", file_name).group(0)
        print(f"Working on {base_name}", flush = True)
        #Get one file ready 
        shutil.copy(f"{self.file_path}/{file_name}",
                    f"{self.temp_path}")
        if format == "json":
            transform_json(f"{self.temp_path}/{file_name}", drop_keys, drop_abstractless)
        dataset = load_dataset(format, data_files =f"{self.temp_path}/{file_name}")
        print("Dataset is loaded", flush = True)
        dataset = pre_process(dataset)
        #Accessing this value in the dictionary makes it easier to manipulate
        dataset = dataset['train']
        #Removing articles that have missing fields 
        dataset = dataset.filter(lambda ex: ex['item'] is not None)
        dataset = dataset.filter(lambda ex: ex['abstract'] is not None)
        print("Filtered", flush = True)
        #Creating embeddings (may take a long time)
        embeddings_dataset = dataset.map(
            lambda x: {"embeddings": get_embeddings(x["full_text"]).detach().cpu().numpy()[0]},
            num_proc = num_cores)
        #Saving embeddings back to json
        os.makedirs(self.result_path, exist_ok=True)
        if out_format == "json":
            embeddings_dataset.to_json(f"{self.result_path}/emb_{base_name}.json")
        elif out_format == "csv":
            embeddings_dataset.to_csv(f"{self.result_path}/emb_{base_name}.csv")
        #Important to do this to prevent your PC/HPCC from running out of storage
        dataset.cleanup_cache_files()
        os.remove(f"{self.temp_path}/{file_name}")
        
def restore_dict(self, year, base_name):
    '''
    og_dict -- path to the original dictionary
    new_dict -- path to the new embeddings (dictionary)
    '''
    os.makedirs(f'{self.emb_corpus}/{year}', exist_ok = True)
    with open(f"{self.file_path}/{base_name}", "r") as file:
        arts_dict = json.load(file)
    #This might be problematic if the file is too large
    emb_dict = read_json_file(f"{self.result_path}/emb_{base_name}")
    for key in arts_dict:
        if arts_dict[key]['item'] == "" or arts_dict[key]['abstract'] == "":
            continue
        else:
            arts_dict[key]['embeddings'] = copy.deepcopy(emb_dict[key])
    with open(f"{self.emb_corpus}/{year}/{base_name}", "w") as f:
        json.dump(arts_dict, f)
    

def embed_all_wos(self, tokenizer, model, adapter_name, device):
    folders = [os.path.join(self.corpus_path, folder['Path']) for folder in rclone.ls(self.corpus_path)]
    for folder in folders:
        files = [os.path.join(folder, file['Path']) for file in rclone.ls(folder)]
        for file in files:
            base_name = os.path.basename(file)
            year = re.search(r"\d{4}", file).group(0)
            print(f"Working on {file}", flush = True)
            shutil.copy(file, self.file_path)
            embed(self, tokenizer, model, adapter_name, device, drop_abstractless=True)
            restore_dict(self, year, base_name)
            os.remove(f"{self.file_path}/{base_name}")
            os.remove(f"{self.result_path}/emb_{base_name}")
            
            
            
        
if __name__ == "__main__":
    #If you hate tqdm, activate the line below
    disable_progress_bar()
    pavel_actor = embedder(file_path = "modal_code/embeddings/file_system/start_files", 
                           temp_path = "modal_code/embeddings/file_system/temp_files", 
                           result_path = "modal_code/embeddings/file_system/result_files", 
                           cache = "modal_code/embeddings/file_system/.cache", 
                           log_file = "modal_code/embeddings/file_system/logs",
                           corpus_path = "modal_code/author_citations/file_system/upd_arts",
                           emb_corpus= "modal_code/embeddings/file_system/embedded_corpus")
    os.makedirs(pavel_actor.log, exist_ok = True)
    logging.basicConfig(filename=f'{pavel_actor.log}/output.log', level=logging.INFO)
    sys.stdout = sys.stderr = open(f'{pavel_actor.log}/output.log', 'a')
    with open(f'{pavel_actor.log}/output.log', 'w'):
        pass
    os.environ['HF_HOME'] = pavel_actor.cache
    #Very important (these should be called before 5th stage runs)
    set_start_method("spawn")
    device = torch.device("cuda")
    tokenizer, model, adapter_name = init_models()
    embed_all_wos(pavel_actor, tokenizer, model, adapter_name, device)
    print("Done")
