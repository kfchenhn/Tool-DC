import pandas as pd
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from rank_bm25 import BM25Okapi
from src.llm import ChatGpt
import json
import csv
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# create_ada_embedding(corpus_cid_dict,tool_emb_file,type='save')
def create_ada_embedding(corpus_cid_dict,output_path,type='save'):
  
    
    batch_size = 4
    if type=='save':
        if os.path.exists(output_path):
            print(f'load embedding {output_path}')
            with open(output_path, 'r') as f:
                key_emb_dict = json.load(f)

        if not os.path.exists(output_path):
            text_list = list(corpus_cid_dict.keys())
            cid_list = list(corpus_cid_dict.values())

            key_emb_dict = {}
            model =  ChatGpt("text-embedding-ada-002")
            print(f'start to generate embedding {output_path}')
                    
            for i in tqdm(range(0,len(text_list),batch_size)):
                batch_text = text_list[i:min(i+batch_size,len(text_list))]
                model.dataset_generate(batch_text,key_emb_dict)

                with open(output_path, 'w') as f:
                    json.dump(key_emb_dict, f,indent=4)
        
    else:
        key_emb_dict = {}
        text_list = list(corpus_cid_dict.keys())
        cid_list = list(corpus_cid_dict.values())
        model =  ChatGpt("text-embedding-ada-002")
        print("生成embedding....")
        for i in tqdm(range(0,len(text_list),batch_size)):
            batch_query = text_list[i:min(i+batch_size,len(text_list))]
            model.dataset_generate(batch_query,key_emb_dict)
    
    key_list = list(key_emb_dict.keys())
    # print("输出corpus_cid_dict：")
    # print(corpus_cid_dict)
    cor_list = list(corpus_cid_dict.keys())
    id_corpus_emb_dict = {}
    for cor in cor_list:
        if cor in key_list:
            id_corpus_emb_dict[corpus_cid_dict[cor]] = key_emb_dict[cor]
        else:
            id_corpus_emb_dict[corpus_cid_dict[cor]] = [0]*len(list(key_emb_dict.values())[0])
            # continue

    return id_corpus_emb_dict

def process_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))
        corpus2tool[(doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))] = doc['category_name'] + '\t' + doc['tool_name'] + '\t' + doc['api_name']
    return ir_corpus, corpus2tool
def read_query(data_path):
    queries_df = pd.read_csv(data_path, quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
    query_list = []
    qid_list = []
    for row in queries_df.itertuples():
        query_list.append(row.query)
        qid_list.append(row.qid)
    return qid_list,query_list

class TF_IDF_Model(object):
    def __init__(self, corpus):
        self.documents_number = len(corpus)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))
        
    def get_score(self, query):
      
        # 将查询转换为TF-IDF向量
        query_vector = self.vectorizer.transform([query])
        
        # 计算查询与每个文档的余弦相似度
        cosine_similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()

        return cosine_similarities


    

def initialize_retriever(model_path,retriever_type,output_dir,phase):
    """
    ir_corpus: {tool_id:content}
    ir_relevant_docs:{qid: (tool_id)}
    id_corpus_emb_dict: {tool_id:embedding}
    """
    
    bm25,tf_idf,id_corpus_emb_dict,retriever_model=None,None,None,None
    ir_relevant_docs = {}
    ir_corpus = {}
    documents_df = pd.read_csv(os.path.join(output_dir,'corpus.tsv'), sep='\t')
    for row in documents_df.itertuples():
        ir_corpus[int(row.docid)] = row.document_content
    #####
    # ir_corpus, _ = process_retrieval_ducoment(documents_df)
    
    labels_df = pd.read_csv(os.path.join(output_dir, f'qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)
    corpus_ids = list(ir_corpus.keys())
    corpus = [ir_corpus[cid] for cid in corpus_ids]
    if retriever_type == 'bm25':
        tokenized_tools = []
        for tool in corpus:
            tokenized_tools.append(tool.split())
        bm25 = BM25Okapi(tokenized_tools)
    elif retriever_type == 'TF-IDF':
        tf_idf = TF_IDF_Model(corpus)
            
    elif retriever_type == 'domain_bert' or retriever_type == 'ToolRetriever' or retriever_type == 'miniLM' or retriever_type == 'hybird_retr' or retriever_type == 'reranker':   

        tokenized_tools = []
        for tool in corpus:
            tokenized_tools.append(tool.split())
        bm25 = BM25Okapi(tokenized_tools)

        word_embedding_model = models.Transformer(model_path, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        retriever_model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device='cuda' if torch.cuda.is_available() else 'cpu')
        corpus_embeddings_list = retriever_model.encode(
            corpus,
            batch_size=512,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        id_corpus_emb_dict = dict(zip(corpus_ids, corpus_embeddings_list))        

    
    elif retriever_type == 'ada': 
        
        # if "ada" not in output_dir:
        tool_emb_file = os.path.join(output_dir,"ada/tool_emb.json")
        # else:
        #     tool_emb_file = os.path.join(output_dir,"tool_emb.json")
        corpus_cid_dict = {content:cid for cid,content in zip(corpus_ids,corpus)}
        

        id_corpus_emb_dict = create_ada_embedding(corpus_cid_dict,tool_emb_file,type='save')
    
    """
    ir_corpus: {tool_id:content}
    ir_relevant_docs:{qid: (tool_id)}
    id_corpus_emb_dict: {tool_id:embedding}
    """
    return ir_corpus,ir_relevant_docs,id_corpus_emb_dict,retriever_model,bm25,tf_idf

def batch_read_json(filename, batch_size):

    current_batch = []
    with open(filename, 'rb') as f:
        parser = json.items(f, 'item')
        for item in parser:
          
            processed_item = {
                'id': item['id'],
                'orig_output': item['orig_output'],
                'instruction': item['instruction'],
                'new_output': item['new_output']
            }
            current_batch.append(processed_item)
            
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

    if current_batch:
        yield current_batch