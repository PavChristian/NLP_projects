import json, copy, os, logging, sys, re, torch
import pandas as pd
from rclone_python import rclone
import chromadb, importlib.util, math
import numpy as np
import xlsxwriter, itertools
from multiprocess import set_start_method
from pyspark.sql import functions as F
from pyspark.sql import types as T

'''Code for creating custom samples and running tests on them'''

class sampler:
    def __init__(self, corpus_path, temp_folder, json_folder, excel_files,
                 csv_folder, embedder_path, emb_files,
                 chroma_path, cit_path, misc_dicts, auth_folders,
                 log_path = "modal_code/sampler/logs"):
        #Path to wos corpus
        #Used for pulling readily available author- and article-level metadata 
        self.corpus_path = corpus_path
        #Folder for temp files 
        self.temp_folder = temp_folder
        #Folder for final jsons
        self.json_folder = json_folder 
        #Folder for final csvs 
        self.csv_folder = csv_folder
        #Folder for final excel files 
        self.excel_files = excel_files
        #Path to the embedder file 
        self.embedder = embedder_path 
        #Where embedding happens
        self.emb_files = emb_files
        #Path to the chroma DB 
        self.chroma = chroma_path
        #Path to citations 
        self.cit_path = cit_path
        #Dictionaries needed for checking the pressence of journals, conferences, etc. in WOS data
        self.misc_dicts = misc_dicts
        #Updated author folders 
        self.auth_folders = auth_folders
        #Folder for logs 
        self.log = log_path
        

def lower_list(input_list):
    if input_list is None:
        return []
    else:
        return [elem.lower() for elem in input_list]

def find_right_names(self, name_list):
    '''
    self -- class defined above 
    name_list -- name of journals or conferences to check
    '''
    #Load in the list of journals
    #Names of files come from the post_processing class
    # -- journal_dir.json -- journal names
    missing_names = []
    with open(f"{self.misc_dicts}/journal_dir.json") as f:
        data = json.load(f)
    list_of_names = lower_list(list(data.keys()))
    list_of_names = [name.split("_")[0] for name in list_of_names]
    #Clear out NAs
    name_list = [name.lower().strip() for name in name_list if isinstance(name, str)]
    for name in name_list:
        if name not in list_of_names:
            missing_names.append(name)
    #Return missing names (they have not been matched) for two reasons:
    #1) Journal name is not in the sample (WoS does not have them)
    #2) Journal name supplied is different from that in Wos 
    return missing_names
    
    
    
        

def should_include_entry(entry, auth_ids=None, jour_list=None, field_list=None, pub_list=None, conf_list=None):
    """Check if an entry should be included based on various filters
    Author_ids -- supply a list of researcher ids that should be in the sample.
    Jour_list -- list of journals that should be in the sample
    NOTE: Clarivate journal headings may be in different format from the standard journal names -- they mave typos, spaces, use different symbols, and skip conjunctions
    field_list -- supply a list of fields (using the standard wOS classification)
    pub_list -- supply a list of pub names (places where a paper was published)
    conf_list -- supply a list of conferences 
    """
    
    '''Use lower case letters to avoid punctuation issues'''
    def in_list_or_none(item, lst):
        return lst is None or item in lst

    def in_lower_list_or_none(item, lst):
        return lst is None or item.lower() in lower_list(lst)

    # Check if at least one condition is true
    if auth_ids is not None:
        if any(r_id in auth_ids and r_id != '' for r_id in entry.get('r_ids', [])):
            return True

    if jour_list is not None:
        if 'source' in entry and entry['source'].lower() in lower_list(jour_list):
            return True

    if field_list is not None:
        fields = entry.get('subject_traditional')
        for field in fields:
            if field in field_list:
                return True

    if pub_list is not None:
        if entry.get('p_display_name').lower() in lower_list(pub_list):
            return True

    if conf_list is not None:
        conf_info = entry.get('conf_info')
        if conf_info is not None:
            if any(confy in conf['conf_title'].lower() for conf in conf_info for confy in lower_list(conf_list)):
                return True

    return False



def should_include_entry_pyspark(entry, auth_ids=None, jour_list=None, field_list=None, pub_list=None, conf_list=None):
    """
    Same function but in PySpark
    """

    def in_list_or_none(item, lst):
        return lst is None or item in lst

    def in_lower_list_or_none(item, lst):
        return lst is None or item.lower() in [x.lower() for x in lst]

    def lower_list(lst):
        return [x.lower() for x in lst]

    # Function to be used in the Spark UDF
    def include(entry):
        if auth_ids is not None:
            if any(r_id in auth_ids and r_id != '' for r_id in entry.get('r_ids', [])):
                return True

        if jour_list is not None:
            if 'source' in entry and entry['source'].lower() in lower_list(jour_list):
                return True

        if field_list is not None:
            fields = entry.get('subject_traditional', [])
            for field in fields:
                if field in field_list:
                    return True

        if pub_list is not None:
            if entry.get('p_display_name', '').lower() in lower_list(pub_list):
                return True

        if conf_list is not None:
            conf_info = entry.get('conf_info', [])
            for conf in conf_info:
                if any(confy in conf['conf_title'].lower() for confy in lower_list(conf_list)):
                    return True

        return False

    # Register the function as a UDF
    include_udf = F.udf(include, T.BooleanType())
    
    return include_udf(entry)


#Obtain a sample of articles with required specifications 
def obtain_sample(self, start_year, end_year, jour_list = None,
                  field_list = None, pub_list = None, auth_ids = None,
                  conf_list = None):
    '''start_year -- the first year from which articles should be pulled 
    end_year -- last year from which articles should be pulled 
    jour_list -- list of journals from which to take articles
    field_list -- list of fields (subject traditional) from which articles should be taken
    pub_list -- list of publishers which published each article
    auth_ids -- list of author ids for authors we would like to examine
    affiliations -- list of affiliations attached to each author record'''
    
    final_dict = {}
    folders = [folder['Path'] for folder in rclone.ls(self.corpus_path) if (start_year <= int(folder['Path']) <= end_year)]
    for folder in folders:
        article_folder = f'{self.corpus_path}/{folder}/articles'
        year = re.search(r"(\d){4}", folder).group(0)
        print(f"Working on {year}", flush = True)
        files =  [os.path.join(article_folder, file['Path']) for file in rclone.ls(article_folder)]
        for file in files:
            with open(file, "r") as json_file:
                art_file = json.load(json_file)
            for key, entry in art_file.items():
                if should_include_entry(entry, auth_ids, jour_list, field_list, pub_list, conf_list):
                    final_dict[key] = copy.deepcopy(entry)
                    final_dict[key]['year'] = year
        print(len(final_dict),flush = True)
            
    return final_dict

def obtain_sample_spark(self, start_year, end_year, jour_list = None,
                  field_list = None, pub_list = None, auth_ids = None,
                  conf_list = None):
    '''
    Same function but in PySpark
    '''
    final_dict = {}
    # Retrieve the list of folders within the corpus path corresponding to the year range
    folders = [folder['Path'] for folder in rclone.ls(self.corpus_path) if (start_year <= int(folder['Path']) <= end_year)]

    for folder in folders:
        article_folder = f'{self.corpus_path}/{folder}/articles'
        year = re.search(r"(\d){4}", folder).group(0)
        print(f"Working on {year}", flush=True)

        # Get the list of article files in the current folder
        files = [os.path.join(article_folder, file['Path']) for file in rclone.ls(article_folder)]

        for file in files:
            # Read the JSON file as a Spark DataFrame
            df = self.spark.read.json(file)

            # Apply the should_include_entry UDF to filter the DataFrame
            should_include_entry_udf = F.udf(lambda entry: should_include_entry(entry, auth_ids, jour_list, field_list, pub_list, conf_list), T.BooleanType())

            filtered_df = df.filter(should_include_entry_udf(F.col('entry')))

            # Collect the filtered results and add them to the final dictionary
            for row in filtered_df.collect():
                entry = row.asDict()
                key = entry.get('key')
                final_dict[key] = copy.deepcopy(entry)
                final_dict[key]['year'] = year

        print(len(final_dict), flush=True)

    return final_dict

def convert_to_csv(data, index_name):
    '''Converts json dicts into a csv
    data -- json file 
    index_name -- what the index column in the out csv should be called'''
    df = pd.DataFrame.from_dict(data, orient='index')
    df.rename_axis(index_name)
    return df



def convert_to_csv_path(path_in, path_out, index):
    data = []
    with open(path_in, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and append to the list
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.set_index(index, inplace = True)
    df.to_csv(path_out)
    
def convert_to_xlsx_path(path_in, path_out, index):
    data = []
    with open(path_in, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and append to the list
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.set_index(index, inplace = True)
    df.to_excel(path_out)


def save_files(self, data, name, df):
    '''saves all files to json and csv'''
    with open(f"{self.json_folder}/{name}.json", "w") as json_file:
        json.dump(data, json_file, indent = 2)
    df.to_csv(f"{self.csv_folder}/{name}.csv")
    
'''Post-processing functions'''


def add_cited_papers(self, start_year, end_year, article_dict):
    '''
    Adding cited papers to the dataset
    '''
    final_dict = {}
    folders = [folder['Path'] for folder in rclone.ls(self.corpus_path) if (start_year <= int(folder['Path']) <= end_year)]
    for folder in folders:
        article_folder = f'{self.corpus_path}/{folder}/refs'
        year = re.search(r"(\d){4}", folder).group(0)
        print(f"Working on {year}", flush = True)
        files =  [os.path.join(article_folder, file['Path']) for file in rclone.ls(article_folder)]
        for file in files:
            with open(file, "r") as json_file:
                dict_file = json.load(json_file)
            for key in article_dict.keys():
                if 'cited_papers' in dict_file[key].keys():
                    continue
                try:
                    article_dict[key]['cited_papers'] = copy.deepcopy(dict_file[key])
                except:
                    pass
                

def make_author_sample(self, start_year, end_year, article_dict):
    '''Add author names to the data (for co-authors) and produce the author_dictionary
    Author dictionary has the following format 
    {id : {year : ..., ...}} contains all r_ids from the selected article sample and the attached metadata
    '''
    author_dict = {}
    folders = [folder['Path'] for folder in rclone.ls(self.auth_folders) if (start_year <= int(folder['Path']) <= end_year)]
    for folder in folders:
        author_folder = f'{self.auth_folders}/{folder}'
        year = re.search(r"(\d){4}", folder).group(0)
        print(f"Working on {year}", flush = True)
        files =  [os.path.join(author_folder, file['Path']) for file in rclone.ls(author_folder)]
        i = 0
        for file in files:
            with open(file, "r") as json_file:
                author_file = json.load(json_file)
            for key in article_dict.keys():
                if article_dict[key]['year'] == folder:
                    if 'aut_names' not in article_dict[key].keys():
                        article_dict[key]['aut_names'] = {}
                    r_ids = article_dict[key]['r_ids']
                    for r_id in r_ids:
                        co_authors = [rid for rid in r_ids if rid != r_id]
                        fake_key = r_id + "_" + year
                        if r_id == '':
                            continue
                        #Adding author data to the original dictionary
                        try:
                            article_dict[key]['aut_names'].update({r_id: author_file[r_id]['wos_standard']})
                            #Filling up the author dictionary
                            
                            if fake_key not in author_dict:
                                author_dict[fake_key] = copy.deepcopy(author_file[r_id])
                                author_dict[fake_key]['sample_ids'] = []
                                #only lists co-authors for this year
                                author_dict[fake_key]['year_co_auths'] = []
                                #maps co-authors to each paper
                                author_dict[fake_key]['paper_author_map'] = {}
                                author_dict[fake_key]['paper_author_names'] = {}
                            #Adding extra papers to authors who wrote them
                            author_dict[fake_key]['sample_ids'].append(key)
                            author_dict[fake_key]['sample_ids'] = copy.deepcopy(list(set(author_dict[fake_key]['sample_ids'])))
                            author_dict[fake_key]['year_co_auths'].extend(co_authors)
                            author_dict[fake_key]['year_co_auths'] = copy.deepcopy(list(set(author_dict[fake_key]['year_co_auths'])))
                            author_dict[fake_key]['paper_author_map'].update({key: co_authors})
                            author_dict[fake_key]['paper_author_names'].update({key: [author_file[ca]['wos_standard'] for ca in co_authors if ca != '']})
                            i += 1 
                        except:
                            pass
        print(i, flush= True)
    for key in author_dict:
        del author_dict[key]['wos_id']
                        
    return article_dict, author_dict

def init_names(author_dict):
    #Gets the name of every author in the sample
    names_dict = {}
    for key in author_dict.keys():
        fake_key = key.split("_")[0]
        if fake_key not in names_dict:
            names_dict[fake_key] = author_dict[key]['wos_standard']
    return names_dict 


def general_coop(d1, names_dict):
    '''Get the dictionary of all collaborations (real) that took place in the sample (any year)
    Input -- author dictionary created by make_author_sample, names dict created by the previous function
    Output -- dictionary where each value is a list of collaborators
    {'r_id1': [{'co_author_id1': name}, {'co_author_id2': name}, etc.]}'''
    
    out_dict = {}
    
    for key in d1.keys():
        fake_key = key.split("_")[0]
        co_authors = copy.deepcopy(d1[key]['year_co_auths'])
        if fake_key not in out_dict.keys():
            out_dict[fake_key] = []
            for c_a in co_authors:
                if c_a == '':
                    continue
                out_dict[fake_key].append({c_a: names_dict[c_a]})
        elif fake_key == "":
            pass
        elif fake_key in out_dict:
            for c_a in copy.deepcopy(co_authors):
                #This is horrible code (Genuinely)
                if c_a == '' or {c_a: names_dict[c_a]} in out_dict[fake_key]:
                    continue
                out_dict[fake_key].append({c_a: names_dict[c_a]})

    '''Outputs everyone's collabs and years they've collabed'''
    return out_dict


#Creates layers of extra co-authors
def deep_network(self, d1, full_dict, exp_name, min_depth = 1, max_depth = 4):
    '''Input -- author dict that came from make_author sample (second output)
    Min_depth -- the first layer (immediate co-authors of the author (key in a dictionary))
    Max_depth -- the final layer of citation network'''
    name_dict = init_names(full_dict)
    ref_dict = general_coop(full_dict, name_dict)
    #Out_dict
    os.makedirs(f"{self.json_folder}/{exp_name}", exist_ok=True)
    '''b_num = 1'''
    def return_names(r_id):
        names = [{r_id: d[next(iter(d.keys()))]} for d in ref_dict[r_id]]
        return names
    def return_ids(r_id):
        ids = [next(iter(d.keys())) for d in ref_dict[r_id]]
        return ids
    for key in d1.keys():
        d1[key] = copy.deepcopy(d1[key])
        fake_key = key.split("_")[0]
        for i in range(min_depth, max_depth + 1):
            if i == 1:
                d1[key][f"ids_level_{i}"] = {fake_key : return_ids(fake_key)}
                d1[key][f"names_level_{i}"] = {fake_key: return_names(fake_key)}
                i += 1
            else:
                j = i - 1
                d1[key][f"ids_level_{i}"] = {}
                d1[key][f"names_level_{i}"] = {}
                #Getting ids
                past_ids = copy.deepcopy(d1[key][f"ids_level_{j}"])
                for r_id1 in past_ids.keys(): #First level
                    for r_id2 in past_ids[r_id1]: #Second level(should be a list)
                        d1[key][f"ids_level_{i}"][r_id2] = copy.deepcopy(return_ids(r_id2))
                        d1[key][f"names_level_{i}"][r_id2] = copy.deepcopy(return_names(r_id2))
                i += 1
                
        '''if len(d2) == 10000:
            print(f"Done with recursion {b_num}", flush = True)
            with open(f"{self.json_folder}/{exp_name}/deep_authors_{b_num}.json", "w") as b_file:
                json.dump(d2, b_file)
            d2 = {}
            b_num += 1'''
    return d1

'''Not a very effective function'''
def split_author_data(self, full_author_dict, exp_name):
    '''If the author file is too large'''
    new_dict = {}
    os.makedirs(f"{self.temp_folder}/{exp_name}/temp_authors", exist_ok=True)
    for r_id in full_author_dict.keys():
        year = r_id.split("_")[1]
        if year not in new_dict:
            new_dict[year] = {}
        new_dict[year][r_id] = copy.deepcopy(full_author_dict[r_id])
    for year_key in new_dict.keys():
        print(f"Working on {year_key}", flush = True)
        with open(f"{self.temp_folder}/{exp_name}/temp_authors/auths_from_{year_key}.json", "w") as hj_file:
            json.dump(new_dict[year_key], hj_file)
    print("Done")

def recurse_over_files(self, full_author_dict, exp_name):
    files = [os.path.join(f"{self.temp_folder}/{exp_name}/temp_authors", file['Path']) for file in rclone.ls(f"{self.temp_folder}/{exp_name}/temp_authors")]
    for file in files:
        #Loading in each year file
        with open(file, "r") as hh:
            auth_data = json.load(hh)
        print(f"Working on {file} of length {len(auth_data)}", flush=True)
        auth_data = deep_network(self, auth_data, full_author_dict, exp_name)
        with open(f"{self.json_folder}/{exp_name}/{os.path.basename(file)}", "w") as b_file:
                json.dump(auth_data, b_file)
                    


'''Embeddings'''

def import_module_from_path(module_name, file_path):
    '''For importing embedding functions as a module'''
    # Ensure the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

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


def split_dict(d, chunk_size=60000):
    """
    Splits a dictionary into multiple dictionaries, each with a maximum length of chunk_size.
    
    Parameters:
        d (dict): The original dictionary to be split.
        chunk_size (int): The maximum size for each sub-dictionary. Default is 60000.
        
    Returns:
        list: A list of dictionaries, each with a length of at most chunk_size.
    """
    keys = list(d.keys())
    return [{k: d[k] for k in keys[i:i + chunk_size]} for i in range(0, len(keys), chunk_size)]

def create_neighbors_all(self, trial_name, article_dict = None, nns = 16, delete_dicts = False, dicts_exist = True):
    '''
    Input -- article dictionary (needs to have embeddings)
    Output -- dictionary (ies) with top-n nearest neighbors, including their titles
    delete_dicts -- False if you do not want to update your collections and True if you want to delete and repopulate them
    delete_dicts and dicts_exist should never be both True or both False
    dicts_exist -- True if you do not want to create new chroma collections 
    '''
    if article_dict is None:
        with open(f"{pavel_sampler.temp_folder}/{trial_name}/arts_ver2.json", "r") as ad:
            article_dict = json.load(ad)
    client = chromadb.PersistentClient(path=self.chroma)
    #Deleting collections is only necessary if something looks wrong below 
    #Chroma is bugged sometimes and does not let in updates (to embeddings, metadata, etc)
    if delete_dicts:
        try:
            client.delete_collection("test")
            client.delete_collection("test2")
            client.delete_collection("test1")
        except:
            pass
    #Obtain or create collection 
    #Each collection can pnly have one underlying distance function
    if not dicts_exist:
        collection = client.get_or_create_collection(name="test",
                                                    metadata={"hnsw:space": "cosine"})
        collection2 = client.get_or_create_collection(name="test1",
                                                    metadata={"hnsw:space": "l2"})
        collection3 = client.get_or_create_collection(name="test2",
                                                    metadata={"hnsw:space": "ip"})
    #Fields 
    dicts = split_dict(article_dict)
    del article_dict
    for article_dict in dicts:
        documents = [article_dict[key]['item'] + "\n" + article_dict[key]['abstract'] for key in article_dict.keys()]
        ids = [key for key in list(article_dict.keys())]
        embeddings = [article_dict[key]['embeddings'] for key in article_dict.keys()]
        #Not adding a lot of metadata right now since it is not necessary
        metadata = [{"title": article_dict[key]['item'], "ids": key} for key in article_dict.keys()]
        '''
        Chroma does not let you load in more than 60K items through one update statement
        '''
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas= metadata
        )
        collection2.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas= metadata
        )
        collection3.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas= metadata
        )
    print("Collection fully updated", flush = True)
    i = 0
    #Writing the file with updated embeddings
    with open(f"{self.json_folder}/{trial_name}/arts_ver3.json", "w") as out_file:
        for article_dict in dicts:
            for key in article_dict.keys():
                one_line_dict = {}
                one_line_dict['wos_id'] = key
                one_line_dict.update(copy.deepcopy(article_dict[key]))
                #Creating cosine distances
                nns1 = collection.query(query_embeddings=article_dict[key]['embeddings'],
                            n_results = nns,
                            include = ['distances', 'metadatas'],
                            #The statement below prevents the query search from pulling the article on which the query is made
                            where = {'ids' : {"$ne": key}})
                one_line_dict['cosine_nns'] = copy.deepcopy(nns1['ids'][0])
                one_line_dict['cosine_distances'] = copy.deepcopy(nns1['distances'][0]) 
                one_line_dict['cosine_titles'] = copy.deepcopy(nns1['metadatas'][0])
                #Creating squared l2 distances
                nns2 = collection2.query(query_embeddings=article_dict[key]['embeddings'],
                            n_results = nns,
                            include = ['distances', 'metadatas'],
                            where = {'ids' : {"$ne": key}})
                one_line_dict['l2_nns'] = copy.deepcopy(nns2['ids'][0])
                one_line_dict['l2_distances'] = copy.deepcopy(nns2['distances'][0]) 
                one_line_dict['l2_titles'] = copy.deepcopy(nns2['metadatas'][0])
                #Creating inner product distances
                nns3 = collection3.query(query_embeddings=article_dict[key]['embeddings'],
                            n_results = nns,
                            include = ['distances', 'metadatas'],
                            where = {'ids' : {"$ne": key}})
                one_line_dict['ip_nns'] = copy.deepcopy(nns3['ids'][0])
                one_line_dict['ip_distances'] = copy.deepcopy(nns3['distances'][0]) 
                one_line_dict['ip_titles'] = copy.deepcopy(nns3['metadatas'][0])
                #Tracking iteration progress 
                if i % 10000 == 0:
                    print(f"{i} have been done")
                

                i += 1
                json_line = json.dumps(one_line_dict)
                out_file.write(json_line + '\n')
                
                
def create_neighbors_one(self, trial_name, collection_name, dist,
                         article_dict = None, nns = 16, delete_col = False, cols_exist = True):
    '''
    Input -- article dictionary (needs to have embeddings)
    Output -- dictionary (ies) with top-n nearest neighbors, including their titles
    delete_dicts -- False if you do not want to update your collections and True if you want to delete and repopulate them
    delete_dicts and dicts_exist should never be both True or both False
    dicts_exist -- True if you do not want to create new chroma collections 
    '''
    if article_dict is None:
        with open(f"{pavel_sampler.temp_folder}/{trial_name}/arts_ver2.json", "r") as ad:
            article_dict = json.load(ad)
    client = chromadb.PersistentClient(path=self.chroma)
    #Deleting collections is only necessary if something looks wrong below 
    #Chroma is bugged sometimes and does not let in updates (to embeddings, metadata, etc)
    if delete_col:
        try:
            client.delete_collection(collection_name)
        except:
            pass
    #Obtain or create collection 
    #Each collection can pnly have one underlying distance function
    if not cols_exist:
        collection = client.get_or_create_collection(name=collection_name,
                                                    metadata={"hnsw:space": dist})
    #Fields 
    dicts = split_dict(article_dict)
    del article_dict
    for article_dict in dicts:
        documents = [article_dict[key]['item'] + "\n" + article_dict[key]['abstract'] for key in article_dict.keys()]
        ids = [key for key in list(article_dict.keys())]
        embeddings = [article_dict[key]['embeddings'] for key in article_dict.keys()]
        #Not adding a lot of metadata right now since it is not necessary
        metadata = [{"title": article_dict[key]['item'], "ids": key} for key in article_dict.keys()]
        '''
        Chroma does not let you load in more than 60K items through one update statement
        '''
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas= metadata
        )
    print("Collection fully updated", flush = True)
    i = 0
    #Writing the file with updated embeddings
    with open(f"{self.json_folder}/{trial_name}/arts_ver3_{dist}.json", "w") as out_file:
        for article_dict in dicts:
            for key in article_dict.keys():
                one_line_dict = {}
                one_line_dict['wos_id'] = key
                one_line_dict.update(copy.deepcopy(article_dict[key]))
                #Creating cosine distances
                nns1 = collection.query(query_embeddings=article_dict[key]['embeddings'],
                            n_results = nns,
                            include = ['distances', 'metadatas'],
                            #The statement below prevents the query search from pulling the article on which the query is made
                            where = {'ids' : {"$ne": key}})
                one_line_dict[f'{dist}_nns'] = copy.deepcopy(nns1['ids'][0])
                one_line_dict[f'{dist}_distances'] = copy.deepcopy(nns1['distances'][0]) 
                one_line_dict[f'{dist}_titles'] = copy.deepcopy(nns1['metadatas'][0])
                #Tracking iteration progress 
                if i % 10000 == 0:
                    print(f"{i} have been done", flush = True)
                i += 1
                json_line = json.dumps(one_line_dict)
                out_file.write(json_line + '\n')
    #Save the final version in other formats
    os.makedirs(f"{self.csv_folder}/{trial_name}", exist_ok= True)
    convert_to_csv_path(f"{self.json_folder}/{trial_name}/arts_ver3_{dist}.json",
                        f"{self.csv_folder}/{trial_name}/arts_ver3_{dist}.csv",
                        "wos_id")
    os.makedirs(f"{self.excel_files}/{trial_name}", exist_ok= True)    
    convert_to_xlsx_path(f"{self.json_folder}/{trial_name}/arts_ver3_{dist}.json",
                        f"{self.excel_files}/{trial_name}/arts_ver3_{dist}.xlsx",
                        "wos_id")
                
def create_dirs(self):
    '''
    Creates dictionaries for embedding files
    '''
    os.makedirs(self.file_path, exist_ok= True)
    os.makedirs(self.result_path, exist_ok= True)
    os.makedirs(self.temp_path, exist_ok= True)
    os.makedirs(self.cache, exist_ok= True)



def restore_arts_dict(self, arts_dict, trial_name):
    '''
    Re-adds metadata back into the original article dictionary 
    self -- the embedder class 
    arts_dict -- the article dictionary with metadata but without embeddings
    trial_name -- name of the trial
    '''
    emb_dict = read_json_file(f"{self.result_path}/emb_arts_ver1.5.json")
    for key in arts_dict:
        if arts_dict[key]['item'] == "" or arts_dict[key]['abstract'] == "":
            continue
        else:
            arts_dict[key]['embeddings'] = copy.deepcopy(emb_dict[key])
    return arts_dict


def create_clean_sample(arts_dict):
    '''
    Drop articles without abstracts
    '''
    new_dict = {}
    for key in arts_dict:
        if arts_dict[key]['has_abstract'] == "Y":
            new_dict[key] = arts_dict[key]
    return new_dict
        
                


def embedding_pipeline(self, arts_dict, trial_name, device, drop_abstractless = True):
    '''
    self -- the main class used 
    arts_dict -- article dictionary that will be embedded 
    trial_name -- name of the trial 
    '''
    my_module = import_module_from_path('embedder', 
                                        self.embedder)
    #Drop articles without abstracts
    if drop_abstractless:
        arts_dict = create_clean_sample(arts_dict)

    #Initializing the embedder class
    pavel_embedder = my_module.embedder(
        #Get the needed file to this folder
        file_path = f"modal_code/sampler/emb_files/start_files/{trial_name}",
                                        temp_path = "modal_code/sampler/emb_files/temp_files",
                                        result_path = f"modal_code/sampler/emb_files/results/{trial_name}",
                                        cache = 'modal_code/sampler/emb_files/.cache',
                                        log_file = None)
    #Create dirs
    create_dirs(pavel_embedder)
    #Save the file into the embedding folder

    with open(f"{pavel_embedder.file_path}/arts_ver1.5.json", "w") as og_file:
        json.dump(arts_dict, og_file)
    os.environ['HF_HOME'] = pavel_embedder.cache
    
    tokenizer, model, adapter_name = my_module.init_models()
    #Place the necessary file here before starting the process
    #NOTE: this function strips the original article dictionary of many of the metadata fields
    my_module.embed(pavel_embedder, tokenizer, model, 
                    adapter_name, device, out_format = "json")
    arts_dict = restore_arts_dict(pavel_embedder, arts_dict, trial_name)
    #Returns the final version with embeddings
    with open(f"{pavel_sampler.temp_folder}/{trial_name}/arts_ver2.json", "w") as e_file:
        json.dump(arts_dict, e_file)
    
    
'''Misc post-processing functions'''

def check_unique_levels(file, field, index):
    format = re.search(r"[a-z]{3,4}$", file).group(0)
    if format == "json":
        with open(file, "r") as json_file:
            data = json.load(json_file)
        df = convert_to_csv(data, index)
    elif format == "csv":
        df = pd.read_csv(file)
    
    unique_values = df[field].unique()

    return unique_values, len(unique_values)


def add_yearly_citations(path_to_cit, article_dict): #Articles 
    '''
    Adds yearly citations to each article
    Input 1 -- Path to the folder with total citations per paper 
    Input 2 -- Article dictionary
    Output -- article dictionary with added yearly citations (e.g {1980: 232, 1981: 3232, etc..})
    '''
    files = [os.path.join(path_to_cit, file['Path']) for file in rclone.ls(path_to_cit)]
    for file in files:
        year = re.search(r"(\d){4}", file).group(0)
        print(f"Working on {file}", flush = True)
        with open(file, "r") as jj:
            cit_data = json.load(jj)
        for wos_id in article_dict.keys():
            if article_dict[wos_id]['year'] == year:
                article_dict[wos_id]['yearly_cits'] = copy.deepcopy(cit_data[wos_id])
    return article_dict

def add_yearly_citations_author(path_to_cit, author_dict, start_year, end_year):
    '''
    Adds yearly citations to each article
    Input 1 -- Path to the folder with total citations per author
    Input 2 -- Author dictionary
    Output -- Author dictionary with added yearly citations (e.g {1980: 232, 1981: 3232, etc..})
    '''
    i = 0
    files = [os.path.join(path_to_cit, file['Path']) for file in rclone.ls(path_to_cit)]
    for file in files:
        year = int(re.search(r"(\d){4}", file).group(0))
        if year < start_year or year > end_year:
            continue
        print(f"Working on {file}", flush = True)
        with open(file, "r") as jj:
            cit_data = json.load(jj)
        for r_id in author_dict.keys():
            fake_key = r_id.split("_")[0]
            try:
                author_dict[r_id]['yearly_cits'] = copy.deepcopy(cit_data[fake_key])
                i += 1
            except:
                pass
        print(f"{i} authors matched", flush = True)
    return author_dict

def calculate_total_cits(author_dict):
    '''
    Creates a column with total citations
    Input 1 -- dictionary with yearly citations present 
    Output -- author dictionary with total citations over all years
    '''
    for r_id in author_dict.keys():
        author_dict[r_id]['total_citations'] = sum(author_dict[r_id].values())
    return author_dict


def clean_address(key_value):
    add_list = key_value['addresses']
    if add_list == []:
        return key_value
    zip_list = []
    org_list = []
    suborg_list = []
    for i, add_dict in enumerate(add_list):
        if add_dict['organizations'] == "" or add_dict['organizations'] is None:
            continue
        elif len(add_dict['organizations'].split("; ")) > 1:
            org = add_dict['organizations'].split("; ")[1]
        else:
            org = add_dict['organizations'].split("; ")[0]
        suborg = add_dict['suborganizations']
        zipcode = add_dict['zip']
        if org in org_list or org == "":
            continue
        else:
            org_list.append(org)
            zip_list.append(zipcode)
            suborg_list.append(suborg)
    if len(org_list) > 0:
        for i, loc in enumerate(org_list):
            key_value[f'zip_{i}'] = zip_list[i]
            key_value[f'inst_{i}'] = org_list[i]
            key_value[f'suborg_{i}'] = suborg_list[i]       
    return key_value 
        
        
def get_fy(self, main_dict):
    with open(f"{self.misc_dicts}/author_years.json", "r") as ff:
        author_info = json.load(ff)
    for key in main_dict:
        fake_key = key.split("_")[0]
        main_dict[key]['fy'] = copy.deepcopy(min(author_info[fake_key]))
        main_dict[key]['ly'] = copy.deepcopy(max(author_info[fake_key]))
    return main_dict

def post_process(self, author_dict):
    for key in author_dict.keys():
        '''author_dict[key] = clean_address(author_dict[key])'''
        author_dict[key] = get_fy(self, key, author_dict[key])
        '''del author_dict['all_papers']'''
        del author_dict[key]['batch_number']
    return author_dict


def clean_list(mylist):
    '''
    Cleans the input lists for sample trials
    '''
    return [elem.lower() for elem in mylist if isinstance(elem, str)]


'''Saving'''

def split_dataframe(df, chunk_size):
    """Split a DataFrame into chunks of a specific size."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])
    return chunks

def to_excel_with_max_size(df, output_path, max_rows=1048576, max_columns=16384):
    """Write a DataFrame to an Excel file, splitting it into multiple sheets if necessary."""
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    # Check if DataFrame exceeds Excel's row or column limits
    if df.shape[0] > max_rows or df.shape[1] > max_columns:
        # Split DataFrame into smaller chunks
        row_chunks = split_dataframe(df, max_rows)
        for i, chunk in enumerate(row_chunks):
            sheet_name = f'Sheet{i+1}'
            # Further split columns if necessary
            if chunk.shape[1] > max_columns:
                col_chunks = [chunk.iloc[:, j:j + max_columns] for j in range(0, chunk.shape[1], max_columns)]
                for k, col_chunk in enumerate(col_chunks):
                    col_chunk.to_excel(writer, sheet_name=f'{sheet_name}_part{k+1}', index=True)
            else:
                chunk.to_excel(writer, sheet_name=sheet_name, index=True)
    else:
        df.to_excel(writer, sheet_name='Sheet1', index=True)
    
    writer.close()   


def join_two(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            dict2.update({key: dict1[key]})
    return dict2


def save_temp_files(self, trial_name, art_dict = None , auth_dict = None):
    '''
    Save the first version of the temp files 
    These versions have basic metadata but no embeddings (in article dicts) and citations (author dict)
    '''
    if art_dict is not None:
        with open(f"{self.temp_folder}/{trial_name}/arts_ver1.json", "w") as ff:
            json.dump(art_dict, ff)
    if auth_dict is not None:
        with open(f"{self.temp_folder}/{trial_name}/auths_ver1.json", "w") as ff:
            json.dump(auth_dict, ff)


def save_intermediate(self, art_dict, auth_dict, trial_name):
    '''
    Save the intermediate output (articles and authors to make sure they look correct)
    art_dict -- article dictionary 
    auth_dict -- author dictionary 
    art_save_path -- directory for saving articles 
    auth_save_path -- directory for saving authors
    '''
    save_temp_files(self, trial_name, art_dict, auth_dict)
    save_path = f"{self.temp_folder}/{trial_name}"
    art_df = convert_to_csv(art_dict, "wos_id")
    art_df.index.name = "wos_id"
    auth_df = convert_to_csv(auth_dict, "r_id")
    '''art_df = art_df[art_df['doctype'] == "Article"]'''
    auth_df.index.name = "r_id"
    auth_df['year'] = auth_df.index.str.split("_").str[1]
    auth_df.index = auth_df.index.str.split("_").str[0]
    art_df = art_df[art_df['pubtype'] == "Journal"]
    art_df.to_csv(f"{save_path}/arts_ver1.csv")
    to_excel_with_max_size(art_df, f"{save_path}/arts_ver1.xlsx")
    auth_df.to_csv(f"{save_path}/auths_ver1.csv")
    to_excel_with_max_size(auth_df, f"{save_path}/auths_ver1.xlsx")
    


def load_intermediate(self, trial_name):
    '''Loads the intermediate files back'''
    with open(f"{self.temp_folder}/{trial_name}/arts_ver1.json") as af:
        arts_dict = json.load(af)
    with open(f"{self.temp_folder}/{trial_name}/auths_ver1.json") as auf:
        auths_dict = json.load(auf)
    return arts_dict, auths_dict

def subset_n(dict_obj, N):
    '''
    Subset first n elements of a dictionary
    '''
    return dict(itertools.islice(dict_obj.items(), N))
    

def clean_entry_list(entry_list):
    '''
    Clears nas from a string
    '''    
    entry_list = [elem.lower().strip() for elem in entry_list if isinstance(elem, str)]
    return entry_list

def other_code(self):
    ''''''

    
def main(self, trial_name, start_year, end_year, device, journal_list = None,
         conf_list = None, field_list = None, pub_list = None, auth_ids = None ):
    '''
    Creating a sample using custom journals/author ids/conferences/etc.
    Change the arguments in the functions below as needed
    Trial_name -- the name of the trial being run (Could be accounting-top5, physicstop 10)
    '''
    #Create a temporary folder for this trial 
    os.makedirs(f"{self.temp_folder}/{trial_name}", exist_ok = True)
    os.makedirs(f"{self.json_folder}/{trial_name}", exist_ok = True)
    #Read in a table with the needed metadata 
    #In this example coferences and journals are used. Pre-process them to make sure no NAs and/or upper-case letters are present
    if journal_list is not None:
        missing_journals = find_right_names(self, journal_list)
        if len(missing_journals) > 0:
            return f"The following journals are missing {missing_journals}"
    
    #Search here to confirm missing journal name (https://www.webofscience.com/wos/woscc/basic-search)
    #Create a sample using the specifications defined above 
    #Add extra arguments later (pubs, r_ids, etc.) 
    article_dict = obtain_sample(self, start_year = start_year,
                                 end_year = end_year,
                                 jour_list=journal_list,
                                 conf_list= conf_list)
    print("Now matching authors", flush = True)
    article_dict, author_dict = make_author_sample(self, start_year, end_year,
                                                   article_dict)
    #Add first year an author is in the dataset and clean the address line
    #Add yearly citations to the author dictionary
    author_dict = get_fy(self, author_dict)
    print("Post_processing finished", flush = True)
    save_intermediate(self, article_dict, author_dict, trial_name)
    article_dict, author_dict = load_intermediate(self, trial_name)
    #Add create embeddings
    embedding_pipeline(pavel_sampler, article_dict, trial_name, device)
    print("Embedding complete", flush = True)
    create_neighbors_one(pavel_sampler, trial_name, collection_name= "test_one", dist = "cosine",
                         cols_exist = False, delete_col = True)
    
    '''#Create n-author layers 
    #This function splits files into separate years (to avoid memory problems in the future)
    split_author_data(self, author_dict, trial_name)
    #Iterate over the years to create author with n layers for each of them 
    recurse_over_files(self, author_dict, trial_name)'''
    
    return "Sample created"
    

if __name__ == "__main__":
    #For embedding 
    device = torch.device("cuda")
    set_start_method("spawn")
    #Defining the class
    pavel_sampler = sampler(corpus_path = "modal_code/wos/file_system/jsons",
                            temp_folder = 'modal_code/sampler/temp',
                            json_folder ='modal_code/sampler/jsons',
                            csv_folder ='modal_code/sampler/csvs',
                            excel_files ='modal_code/sampler/xlsx',
                            chroma_path = 'modal_code/sampler/chroma',
                            embedder_path = "modal_code/embeddings/create_embs.py",
                            emb_files="modal_code/sampler/emb_files",
                            cit_path= "wos_cits",
                            auth_folders= "modal_code/author_citations/file_system/final_jsons",
                            misc_dicts = "modal_code/post_processing/file_system/wos_misc")
    
    logging.basicConfig(filename=f'{pavel_sampler.log}/output.log', level=logging.INFO)
    sys.stdout = sys.stderr = open(f'{pavel_sampler.log}/output.log', 'a')
    with open(f'{pavel_sampler.log}/output.log', 'w'):
        pass
    '''journals = pd.read_excel("modal_code/sampler/temp/eleng.xlsx")
    journal_list = clean_entry_list(journals['clean_names'].to_list())'''
    '''journal_list = ['Basket weaving digest', "Chocolate review"]'''
    journal_list = clean_entry_list(['Nature Neuroscience', 'Neuron', 'Cell', 'Brain', 'Nature Methods'])
    result = main(pavel_sampler, 'bio14', 2010, 2024, device, journal_list = journal_list)
    print(result)
    
    
    
    
                
                        