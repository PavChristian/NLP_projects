import json, os, sys, logging, re, copy, random
from rclone_python import rclone
random.seed(49)
'''Necessary components for this to run:
1. Article-level dictionaries from every year
2. Citation-level dictionaries from every year
3. Installed rclone
4. Author-level dictionaries from every year

Misc. facts

1. High memory jobs required (see functions below)


OUTPUT:
author_year files with year-filtered citations'''

class citation_counter:
    def __init__(self, corpus_path, 
                 full_yearly_citations, wos_ids, historic_citations,
                 upd_arts, upd_auths, aut_cit, final_auths,
                 final_jsons, final_csvs, log_path = "modal_code/author_citations/logs/out_log.log"):
        #Path to year folders where articles, authors, and refs are stored for each year
        self.corpus_path = corpus_path
        #Path where we store full_yearly citations (i.e how many times each article was cited per year)
        self.fyc = full_yearly_citations
        #Path where we store ids of articles written each year
        self.wos_ids = wos_ids
        #Path where we store article's citations for all years they were cited in
        self.hc = historic_citations
        #Path where we store updated article dictionaries (with all yearly_citations)
        self.upd_arts = upd_arts
        #Path where we store update author records (author-year-no. of citations that papers published in this year got)
        #E.g {r_id: {year: 2015, papers : [wos_id1, wos_id2], cited: [2015: 444, 2016: 432, 2017: 654]}}
        self.upd_auths = upd_auths
        #Author-level citations (for all time)
        self.aut_cit = aut_cit
        #Final authors with iterator (i)
        self.final = final_auths
        #Path where we store files that have the "real network" for each year
        self.mini_aut = final_jsons
        #The real network in a a csv format
        self.final_csvs = final_csvs
        #Logging 
        self.log_path = log_path

def aggregate_citations(ref_dict_folder):
    '''
    NOTE: This function is called by get_total_cits
    ref_dict_folder (str) -- takes in a folder with ref_dicts "{year}/ref_dicts/
    '''
    files = [file['Path'] for file in rclone.ls(ref_dict_folder)]
    year = re.search(r"(\d){4}", ref_dict_folder).group(0)
    #Init the dict to store the results
    out_dict = {}
    #Store wos_ids
    wos_ids = []
    for file in files:
        file_path = os.path.join(ref_dict_folder, file)
        with open(file_path, "r") as json_file:
            ref_dict = json.load(json_file)
        for key in ref_dict.keys():
            cited_works = copy.deepcopy(ref_dict[key])
            for work in cited_works:
                if work in out_dict.keys():
                    out_dict[work][year] += 1 
                else:
                    out_dict[work] = {}
                    out_dict[work][year] = 1

        wos_ids.extend(list(ref_dict.keys()))
    '''Output -- produces one dictionary with citation counts for each article 
    that was cited within that year + ids of papers in the whole year'''
    return out_dict, year, wos_ids

def save_wos_ids(self, year, wos_ids):
    os.makedirs(self.wos_ids, exist_ok = True)
    save_path = f"{self.wos_ids}/wos_ids_from_{year}.json"
    with open(save_path, "w") as w_file:
        json.dump(wos_ids, w_file, indent = 2)     
    
def get_total_cits(self,
                   multi_processing = False, task_id = None):
    '''
    self -- class object
    multi_processing (bool) - should multiprocessing be used?
    task_id (int) -- task id of the folder (only use when running in parallel)
    Saves output from the previous function
    If multiprocessing is True, task_id needs to be supplied
    '''
    folders = [folder['Path'] for folder in rclone.ls(self.corpus_path,
                                                      dirs_only = True)]
    os.makedirs(self.fyc, exist_ok= True)
    if multi_processing:
        my_folder = os.path.join(self.corpus_path, folders[task_id])
        ref_folder = os.path.join(my_folder, "refs")
        print(f"Working on {my_folder}", flush = True)
        out_dict, year, wos_ids = aggregate_citations(ref_folder)
        with open(f'{self.fyc}/total_cits_{year}.json', "w") as json_file:
            json.dump(out_dict, json_file, indent = 2)
        #Save total wos_ids of this year
        save_wos_ids(self, year, wos_ids)
    else:
        for folder in folders:
            my_folder = os.path.join(self.corpus_path, folder)
            ref_folder = os.path.join(my_folder, "refs")
            print(f"Working on {folder}", flush = True)
            out_dict, year, wos_ids = aggregate_citations(ref_folder)
            with open(f'{self.fyc}/total_cits_{year}.json', "w") as json_file:
                json.dump(out_dict, json_file, indent = 2)
            save_wos_ids(self, year, wos_ids)
    print("get_total_cits completed running")
    '''The resulting dictionary has the following format
    {'wos_id': {2015: 333}, {'wos_id1' : {2015: 21}}} where 333, 21 are the total citations in this year'''
    
def get_historic_citations(self, multi_processing = False, task_id = None):
    '''
    self -- class object
    multi_processing (bool) - should multiprocessing be used?
    task_id (int) -- task id of the folder (only use when running in parallel)
    NOTE: Multiprocessing is strongly advised here. It saves quite a lot of time
    Takes in folder with output from the previous function and wos_id files
    Output -- historic citations. Dictionary (one per year) with the following format:
    {wos_id: {2023: 1231}, {2022: 323}, {2021: 3322}, etc.}'''
    hist_dict = {}
    os.makedirs(self.hc, exist_ok= True)
    #Getting unique article ids from article dictionaries 
    if multi_processing:
        files = [os.path.join(self.wos_ids, file['Path']) for file in rclone.ls(self.wos_ids)]
        my_file = files[task_id]
        year = re.search(r"(\d{4})", my_file).group(0)
        with open(my_file, "r") as w_file:
            wos_ids = json.load(w_file)
        for wos_id in wos_ids:
            hist_dict[wos_id] = {}
        files = [os.path.join(self.fyc, folder['Path']) for folder in rclone.ls(self.fyc)]
        for file in files:
            file_year = re.search(r"(\d{4})", file).group(0)
            if int(file_year) < int(year):
                continue
            print(f"Working on {file}", flush = True)
            with open(file, "r") as c_file:
                fyc = json.load(c_file)
            for wos_id in wos_ids:
                try:
                    hist_dict[wos_id].update(fyc[wos_id])
                except:
                    hist_dict[wos_id].update({file_year:0})
        with open(f"{self.hc}/hc_{year}.json", "w") as hc_file:
            json.dump(hist_dict, hc_file)
    else:
        files = [os.path.join(self.wos_ids, file['Path']) for file in rclone.ls(self.wos_ids)]
        for my_file in files:
            year = re.search(r"(\d{4})", my_file).group(0)
            print(f"Working on {my_file}", flush = True)
            with open(my_file, "r") as w_file:
                wos_ids = json.load(w_file)
            for wos_id in wos_ids:
                hist_dict[wos_id] = {}
            files = [os.path.join(self.fyc, folder['Path']) for folder in rclone.ls(self.fyc)]
            for file in files:
                file_year = re.search(r"(\d{4})", file).group(0)
                if int(file_year) < int(year):
                    continue
                print(f"Working on {file}", flush = True)
                with open(file, "r") as c_file:
                    fyc = json.load(c_file)
                for wos_id in wos_ids:
                    try:
                        hist_dict[wos_id].update(fyc[wos_id])
                    except:
                        hist_dict[wos_id].update({file_year:0})
            with open(f"{self.hc}/hc_{year}.json", "w") as hc_file:
                json.dump(hist_dict, hc_file)
    print("get_historical_citations completed running")
                        
def filter_keys(d, req_keys):
    '''
    d (dict) -- dictionary
    req_keys (list) -- list of keys to be kept in the dictionary
    Strip the dictionary of unnecessary keys'''
    return {key: d[key] for key in req_keys if key in d}
    

def get_author_paper_dicts(self):
    '''Gets authors and their papers in one dictionary
    High memory job 
    Better to run as one process (rather than splitting up in multiple years)
    Do not use multiprocessing
    Takes in folder with all the years 
    Output -- dictionary with {r_id: ['paper1', 'paper2', ...]} all papers across the years for all authors'''
    final_author_dict = {}
    folders = [os.path.join(self.corpus_path, folder['Path']) for folder in rclone.ls(self.corpus_path)]
    for folder in folders:
        print(f"Working on {folder}", flush = True)
        folder_path = os.path.join(folder, "authors")
        files = [os.path.join(folder_path, file['Path']) for file in rclone.ls(folder_path)]
        for file in files:
            with open(file, "r") as art_file:
                art_data = json.load(art_file)
            for key in art_data:
                batch_number = random.randint(1, 26)
                '''Generating a number to join the citations back to the original author dictionary'''
                if key not in final_author_dict:
                    art_data[key]['batch_number'] = copy.deepcopy(batch_number)
                    final_author_dict[key] = {}
                    final_author_dict[key]['batch_number'] = batch_number
                    final_author_dict[key]['wos_ids'] = copy.deepcopy(art_data[key]['wos_id'])
                elif key == '':
                    continue
                else:
                    art_data[key]['batch_number'] = copy.deepcopy(final_author_dict[key]['batch_number'])
                    final_author_dict[key]['wos_ids'].extend(copy.deepcopy(art_data[key]['wos_id']))
            with open(file, "w") as af:
                json.dump(art_data, af, indent = 2)
    os.makedirs(self.upd_auths, exist_ok= True)
    #Saves into 26 different files
    for num in range(1, 27):
        temp_dict = {}
        print(f"Working on {num}", flush = True)
        for a in final_author_dict.keys():
                if final_author_dict[a]['batch_number'] == num:
                    temp_dict[a] = copy.deepcopy(final_author_dict[a])
        with open(f"{self.upd_auths}/authors_{num}.json", "w") as gg:
            json.dump(temp_dict, gg)

def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries and sum the values of common keys.

    Args:
    dict1 (dict): The first dictionary.
    dict2 (dict): The second dictionary.

    Returns:
    dict: A new dictionary with summed values for common keys.
    """
    merged_dict = copy.deepcopy(dict1)  # Start with the keys and values of dict1

    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value  # Sum the values for common keys
        else:
            merged_dict[key] = value  # Add unique keys from dict2

    return merged_dict

def read_json(file_path):
    """
    Reads a JSON file where each dictionary is separated by a newline character.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    list: A list of dictionaries read from the file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure that the line is not empty
                data.append(json.loads(line))
    return data

 
                    
def create_author_citations(self, multi_processing = False, task_id = None):
    '''
    self -- class object
    multi_processing (bool) - should multiprocessing be used?
    task_id (int) -- task id of the folder (only use when running in parallel)
    Multi-processing preferred
    Inserts full citations for every author
    '''
    os.makedirs(self.aut_cit, exist_ok = True)
    #Obtain updated files (26 author files)
    author_files = [os.path.join(self.upd_auths, file['Path']) for file in rclone.ls(self.upd_auths)]
    #Obtain citations for each paper for each year
    historic_files = [os.path.join(self.hc, file['Path']) for file in rclone.ls(self.hc)]
    if multi_processing:
        my_file = author_files[task_id] #Going over one of the 27 available files
        print(f"Working on {my_file}", flush = True)
        with open(my_file, "r") as jk:
            author_data = json.load(jk) #Each line is a dict {r_id: {metadata}}
        for author in author_data:
            author_data[author]['yearly_cits'] = {}
        for file in historic_files:
            with open(file, "r") as h_file:
                hist_data = json.load(h_file)
            for author in author_data.keys():
                papers = author_data[author]['wos_ids']
                #Summing up citations for each paper without forgetting the keys
                for paper in papers:
                    try:
                        paper_dict = copy.deepcopy(hist_data[paper])
                    except:
                        continue
                    present_dict = copy.deepcopy(author_data[author]['yearly_cits'])                
                    author_data[author]['yearly_cits'] = merge_dicts(present_dict, paper_dict)
        #This file should still have iterator ids
        base_name = os.path.basename(my_file)
        with open(f"{self.aut_cit}/{base_name}", "w") as rr:
            json.dump(author_data, rr)   
    else: #Avoid scenarios where the loop has to be called
        for my_file in author_files: #Going over one of the 27 available files
            print(f"Working on {my_file}", flush = True)
            with open(my_file, "r") as jk:
                author_data = json.load(jk) #Each line is a dict {r_id: {metadata}}
            for author in author_data:
                author_data[author]['yearly_cits'] = {}
            for file in historic_files:
                with open(file, "r") as h_file:
                    hist_data = json.load(h_file)
                for author in author_data.keys():
                    papers = author_data[author]['wos_ids']
                    #Summing up citations for each paper without forgetting the keys
                    for paper in papers:
                        try:
                            paper_dict = copy.deepcopy(hist_data[paper])
                        except:
                            continue
                        present_dict = copy.deepcopy(author_data[author]['yearly_cits'])                
                        author_data[author]['yearly_cits'] = merge_dicts(present_dict, paper_dict)
            #This file should still have iterator ids
            base_name = os.path.basename(my_file)
            with open(f"{self.aut_cit}/{base_name}", "w") as rr:
                json.dump(author_data, rr)       
            
           
                
            
def produce_final_authors(self, multi_processing = False, task_id = None):
    '''
    Inserting citations into the original author files
    Running this on hamster is not stable (too slow and and not efficient)
    Multi-processing could speed things up considerably:
    Each task id should focus on one year-folder and run through all 27 subdicts
    Range of task_ids is 27 (if on Mercury, assign from 1 to 27)
    
    '''
    folders = [os.path.join(self.corpus_path, f"{folder['Path']}/authors") for folder in rclone.ls(self.corpus_path)]
    aut_cit_files = [os.path.join(self.aut_cit, file['Path']) for file in rclone.ls(self.aut_cit)]
    if multi_processing:
        my_folder = folders[task_id]
        files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
        for file in files:
            year = re.search(r"\d{4}", file).group(0)
            with open(file, "r") as json_file:
                data = json.load(json_file)
            print(f"Working on {file}", flush = True)
            for aut_file in aut_cit_files:
                mini_author_dict = dict()
                with open(aut_file, "r") as autfile:
                    cit_data = json.load(autfile)
                i = int(re.search(r"\d+", aut_file).group(0))
                for key in data.keys():
                    if key not in mini_author_dict:
                        mini_author_dict[key] = {}
                    if data[key]['batch_number'] == i:
                        mini_author_dict[key] = copy.deepcopy(data[key])
                        mini_author_dict[key]['all_papers'] = copy.deepcopy(cit_data[key]['wos_ids'])
                        mini_author_dict[key]['yearly_cit'] = copy.deepcopy(cit_data[key]['yearly_cits'])
                base_name = os.path.basename(file).replace(".json", "")
                os.makedirs(f"{self.mini_aut}/{year}")
                with open(f"{self.mini_aut}/{year}/file_{i}.json", "w") as ff:
                    json.dump(mini_author_dict, ff)
    else:
        for my_folder in folders:
            files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
            for file in files:
                with open(file, "r") as json_file:
                    data = json.load(json_file)
                print(f"Working on {file}", flush = True)
                for aut_file in aut_cit_files:
                    mini_author_dict = dict()
                    with open(aut_file, "r") as autfile:
                        cit_data = json.load(autfile)
                    i = int(re.search(r"\d+", aut_file).group(0))
                    year = re.search(r"\d{4}", file).group(0)
                    for key in data.keys():
                        if key not in mini_author_dict:
                            mini_author_dict[key] = {}
                        if data[key]['batch_number'] == i:
                            mini_author_dict[key] = copy.deepcopy(data[key])
                            mini_author_dict[key]['all_papers'] = copy.deepcopy(cit_data[key]['wos_ids'])
                            mini_author_dict[key]['yearly_cit'] = copy.deepcopy(cit_data[key]['yearly_cits'])
                    base_name = os.path.basename(file).replace(".json", "")
                    os.makedirs(f"{self.mini_aut}/{year}", exist_ok = True)
                    with open(f"{self.mini_aut}/{year}/{base_name}_{i}.json", "w") as ff:
                        json.dump(data, ff)
                        
def clean_empty_files(self):
    '''
    Cleans the folder (removes empty files from the previous function's output)
    '''
    folders = [os.path.join(self.mini_aut, folder['Path']) for folder in rclone.ls(self.mini_aut)]
    for my_folder in folders[41:]:
        name_author_dict = {}
        files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
        year = re.search(r"\d{4}", my_folder).group(0)
        for file in files:
            print(f"Working on {file}", flush = True)
            with open(file, "r") as hh_file:
                author_data = json.load(hh_file)
            if len(author_data) == 0:
                print(file)
                '''os.remove(file)'''
                        

def create_author_wos(self):
    '''
    Creates paper dictionaries for each author/year pair
    Puts them into the 'year_papers' column 
    Saves the updated files into final_jsons
    '''
    folders = [os.path.join(self.mini_aut, folder['Path']) for folder in rclone.ls(self.mini_aut)]
    for my_folder in folders[41:]:
        name_author_dict = {}
        files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
        year = re.search(r"\d{4}", my_folder).group(0)
        for file in files:
            print(f"Working on {file}", flush = True)
            with open(file, "r") as hh_file:
                author_data = json.load(hh_file)
        #Populating the dictionary
            for key in author_data:
                if key not in name_author_dict:
                    name_author_dict[key] = copy.deepcopy(author_data[key]['wos_id'])
                else:
                    name_author_dict[key].extend(copy.deepcopy(author_data[key]['wos_id']))
            for key in author_data.keys():
                author_data[key]['year_papers'] = copy.deepcopy(name_author_dict[key])
            with open(file, "w") as js:
                json.dump(author_data, js)
            


        
            

def create_full_author_years(self, multi_processing = False, task_id = None):
    os.makedirs(self.mini_aut, exist_ok=True)
    '''
    Takes in a wos folder with files and pushes out an author-level json for each year
    Multiprocessing preferred
    High memory
    '''
    folders = [os.path.join(self.mini_aut, folder['Path']) for folder in rclone.ls(self.mini_aut)]
    if multi_processing:
        my_folder = folders[task_id]
        files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
        with open(f"{self.final_auths}/authors_batch_{my_folder}.json", "w") as f:
            for file in files:
                with open(file, "r") as hh_file:
                    author_data = json.load(hh_file)
                for key in author_data.keys():
                    #This avoid duplicate author keys
                    if key not in hold_out:
                        hold_out.append(key)
                    else:
                        continue
                    json_line = json.dumps({key: author_data[key]})
                    f.write(json_line + "\n")
    else:
        for my_folder in folders[8:]:
            hold_out = {}
            files = [os.path.join(my_folder, file['Path']) for file in rclone.ls(my_folder)]
            year = re.search(r"\d{4}", my_folder).group(0)
            with open(f"{self.final}/authors_batch_{year}.json", "w") as f:
                for file in files:
                    print(f"Working on {file}", flush = True)
                    with open(file, "r") as hh_file:
                        author_data = json.load(hh_file)
                    for key in author_data.keys():
                        #This avoid duplicate author keys
                        try:
                            hold_out[key]
                            continue 
                        except:
                            json_line = json.dumps({key: author_data[key]})
                            f.write(json_line + "\n")
                            hold_out[key] = ""


def create_upd_arts(self, multi_processing = False, task_id = None):
    '''creates updated articles with comprehensive citation data for each one of them
    input 1 (through the self argument) -- wos article records (clean)
    input 2 (through the self argument) -- historical citation data (over all the years)
    output -- wos article records with citation data'''
    if multi_processing:
        iterator = range(task_id, task_id + 1)
    else:
        iterator = range(45)
    wos_folders = [folder['Path'] for folder in rclone.ls(self.corpus_path)]
    hist_citations = [os.path.join(self.hc, file['Path']) for file in rclone.ls(self.hc)]
    for i in iterator:
        my_folder = wos_folders[i]
        print(f"Working on {my_folder}", flush = True)
        my_hist_file = [fold for fold in hist_citations if my_folder in fold][0]
        with open(my_hist_file, "r") as js_file:
            hist_cits = json.load(js_file)
        my_folder_path = f"{self.corpus_path}/{my_folder}/articles"
        article_files = [os.path.join(my_folder_path, file['Path']) for file in rclone.ls(my_folder_path)]
        os.makedirs(f"{self.upd_arts}/{my_folder}", exist_ok = True)
       
        for file in article_files:
            base_file = os.path.basename(file)
            with open(file, "r") as json_file:
                article_data = json.load(json_file)
            for wos_id in article_data.keys():
                if wos_id == "":
                    continue
                article_data[wos_id]['yearly_cit'] = copy.deepcopy(hist_cits[wos_id])
            with open(f"{self.upd_arts}/{my_folder}/upd_{base_file}", "w") as u_file:
                json.dump(article_data, u_file, indent = 2)
                

        
    

def citer_main_process(self):
    '''
    All functions arranged in the order of execution
    '''
    get_total_cits(self)
    get_historic_citations(self)
    get_author_paper_dicts(self)
    #This does not need any prereqs to run 
    create_author_citations(self)
    #Supply historic files and updated authors
    #NOTE:Run the function below on Mercury
    #--------
    produce_final_authors(self)
    #--------
    create_full_author_years(self)
    create_upd_arts(self)
    create_author_wos(self)
    
    


if __name__ == "__main__":
    mega_citer = citation_counter(corpus_path = "modal_code/wos/file_system/jsons",
                                  full_yearly_citations = "modal_code/author_citations/file_system/fyc",
                                  wos_ids = "modal_code/author_citations/file_system/wos_ids",
                                  historic_citations = "modal_code/author_citations/file_system/hist_ci",
                                  upd_arts = "modal_code/author_citations/file_system/upd_arts",
                                  upd_auths = "modal_code/author_citations/file_system/upd_auths",
                                  aut_cit= "modal_code/author_citations/file_system/au_ci",
                                  final_auths="modal_code/author_citations/file_system/author_years",
                                  final_jsons="modal_code/author_citations/file_system/final_jsons",
                                  final_csvs = None)
    logging.basicConfig(filename=mega_citer.log_path, level=logging.INFO)
    sys.stdout = sys.stderr = open(mega_citer.log_path, 'a')
    with open(mega_citer.log_path, 'w'):
        pass 
    '''citer_main_process(mega_citer)'''
    create_upd_arts(mega_citer)
    print("Done", flush = True)


    
    
    
                        
                        