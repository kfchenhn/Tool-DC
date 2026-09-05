from sklearn.metrics import ndcg_score
import numpy as np
import logging
import os
from typing import List, Dict, Set
from tqdm import trange
from tqdm import tqdm
import torch
from multiprocessing import Pool
import heapq
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim
import os
import threading
import torch
import multiprocessing
import torch.nn.functional as F
from src.utils import create_ada_embedding
import json
import copy
from FlagEmbedding import FlagReranker
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class APIEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document.
    """

    def __init__(
        self,
        queries_list: List[str], # query
        queries_id_list: List[str], # qid
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]], 
        id_corpus_emb_dict=None,  # cid => emb
        model=None,
        bm25=None,
        tf_idf=None,
        retriever_type=None,
        output_path=None,
        write_csv=True,
        phase= 'test'
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.queries_id = queries_id_list
        self.queries_list = queries_list
        self.cid_corpus_dict = corpus
        self.corpus_ids = list(corpus.keys())
        self.corpus_list = [corpus[cid] for cid in self.corpus_ids]
      
        self.cid_to_idx = {cid: idx for idx,cid in enumerate(self.corpus_ids)}
        self.relevant_docs = relevant_docs
        self.score_function = cos_sim      
        self.id_corpus_emb_dict = id_corpus_emb_dict
        self.retriever_type = retriever_type
        self.output_path = output_path  
        self.write_csv = write_csv
        self.phase = phase
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # if self.phase == 'test':
        if self.retriever_type == 'bm25':
            self.bm25 = bm25
        elif self.retriever_type == 'TF-IDF':
            self.tf_idf = tf_idf
        
        elif self.retriever_type == 'colbert':
            pass
         
        elif self.retriever_type == 'hybird_retr':
            
            self.bm25 = bm25
            self.model = model
            self.query_embeddings = self.model.encode(
                self.queries_list,
                show_progress_bar=False,
                batch_size=10,
                convert_to_tensor=True,
                device=self.device
            )
        
            with torch.no_grad():
                corpus_ids = []
                corpus_vals = []
                for cid, emb in self.id_corpus_emb_dict.items():
                    corpus_ids.append(cid)
                    if isinstance(emb, torch.Tensor):
                        emb_cpu = emb.cpu().numpy()  
                    elif isinstance(emb, (np.ndarray, list)):
                        emb_cpu = np.array(emb)
                    else:
                        raise TypeError(f"Unsupported embedding type: {type(emb)}")
                        
                    corpus_vals.append(emb_cpu)  

                stacked_embeddings = np.stack(corpus_vals, axis=0) 
                self.corpus_embeddings = torch.tensor(
                    stacked_embeddings,
                    dtype=torch.float32,
                    device=self.device
                )
                self.corpus_ids = corpus_ids
                self.corpus_list = [corpus[cid] for cid in self.corpus_ids]


          


        else:
            if self.retriever_type in ["domain_bert",'ToolRetriever',"reranker"]:
                self.model = model
                self.query_embeddings = self.model.encode(
                    self.queries_list,
                    show_progress_bar=False,
                    batch_size=10,
                    convert_to_tensor=True,
                    device=self.device
                )
                if self.retriever_type == "reranker":
                    self.reranker = FlagReranker('/models/bge-reranker-v2-m3', use_fp16=True)


            elif self.retriever_type == "ada":
                self.query_qid_dict = {query:int(qid) for query,qid in zip(self.queries_list,self.queries_id)}
                if not self.write_csv:
                    self.qid_query_embeddings_dict = create_ada_embedding(self.query_qid_dict,output_path=None,type='no_save')
                else:
                    self.qid_query_embeddings_dict = create_ada_embedding(self.query_qid_dict,output_path=f'{self.output_path}/{self.phase}_ada_emb.json',type='save')
                self.query_embeddings = list(self.qid_query_embeddings_dict.values())       
                if not isinstance(self.query_embeddings, torch.Tensor):
                    self.query_embeddings = torch.tensor(self.query_embeddings, dtype=torch.float32)
                    self.query_embeddings = self.query_embeddings.to(self.device)
                self.queries_id = [self.query_qid_dict[query] for query in self.queries_list]

            with torch.no_grad():
                corpus_ids = []
                corpus_vals = []
                for cid, emb in self.id_corpus_emb_dict.items():
                    corpus_ids.append(cid)
                    if isinstance(emb, torch.Tensor):
                        emb_cpu = emb.cpu().numpy()  
                    elif isinstance(emb, (np.ndarray, list)):
                        emb_cpu = np.array(emb)
                    else:
                        raise TypeError(f"Unsupported embedding type: {type(emb)}")
                        
                    corpus_vals.append(emb_cpu)  

                stacked_embeddings = np.stack(corpus_vals, axis=0) 
                self.corpus_embeddings = torch.tensor(
                    stacked_embeddings,
                    dtype=torch.float32,
                    device=self.device
                )
                self.corpus_ids = corpus_ids
                self.corpus_list = [corpus[cid] for cid in self.corpus_ids]

                
    def __call__(self,model=None,epoch=-1,steps=-1,*args, **kwargs) :


        queries_result_list = [[] for _ in range(len(self.queries_list))]
        scores_list = []
      

        if self.retriever_type in ['domain_bert','ada','ToolRetriever']:
            with torch.no_grad():
                all_scores = self.score_function(self.query_embeddings, self.corpus_embeddings)
                all_scores = all_scores.cpu().numpy()
                sorted_indices = np.argsort(-all_scores, axis=1)
                for query_idx in tqdm(range(len(self.queries_list))):
                    query_sorted_indices = sorted_indices[query_idx] 
                    queries_result_list[query_idx] = [
                        {
                            "corpus_id": self.corpus_ids[doc_idx],
                            "score": float(all_scores[query_idx][doc_idx])
                        }
                        for doc_idx in query_sorted_indices
                    ]          

        elif self.retriever_type == 'bm25':
            for query_idx, query in tqdm(enumerate(self.queries_list)):
                tokenized_query = query.split()
                doc_scores = self.bm25.get_scores(tokenized_query)
                for doc_idx, score in enumerate(doc_scores):
                    corpus_id = self.corpus_ids[doc_idx]
                    queries_result_list[query_idx].append({
                        "corpus_id": corpus_id,
                        "score": round(float(score),2)
                    })
        elif self.retriever_type == 'TF-IDF':
            for query_idx, query in tqdm(enumerate(self.queries_list)):
                doc_scores = self.tf_idf.get_score(query)
                for doc_idx, score in enumerate(doc_scores):
                    corpus_id = self.corpus_ids[doc_idx]
                    queries_result_list[query_idx].append({
                        "corpus_id": corpus_id,
                        "score": round(float(score),2)
                    })
        
        
        elif self.retriever_type == 'colbert':

            for query_idx, query in tqdm(enumerate(self.queries_list)):
                queries_embeddings = self.colbert_model.encode(
                    [query],
                    batch_size=32,
                    is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
                    show_progress_bar=False,
                )
                # Step 3: Retrieve top-k documents
                doc_scores = self.colbert_retriever.retrieve(
                    queries_embeddings=queries_embeddings,
                    k=100000,  # Retrieve the top 10 matches for each query
                )
                
                # doc_scores = self.tf_idf.get_score(query)
                for score_item in doc_scores[0]:
                    
                    corpus_id = score_item["id"]
                    score = score_item["score"]
                    queries_result_list[query_idx].append({
                        "corpus_id": corpus_id,
                        "score": round(float(score),2)
                    })
        
        elif self.retriever_type == 'hybird_retr':
           
            toolretriever_queries_result_list = copy.deepcopy(queries_result_list)
            bm25_queries_result_list = copy.deepcopy(queries_result_list)
          
            with torch.no_grad():
                all_scores = self.score_function(self.query_embeddings, self.corpus_embeddings)
                all_scores = all_scores.cpu().numpy()
                sorted_indices = np.argsort(-all_scores, axis=1)
                for query_idx, query in tqdm(enumerate(self.queries_list)):

               
                    tokenized_query = query.split()
                    doc_scores = self.bm25.get_scores(tokenized_query)
                    for doc_idx, score in enumerate(doc_scores):
                        corpus_id = self.corpus_ids[doc_idx]
                        bm25_queries_result_list[query_idx].append({
                            "corpus_id": corpus_id,
                            "score": round(float(score),2)
                        })


                    query_sorted_indices = sorted_indices[query_idx] 
                    toolretriever_queries_result_list[query_idx] = [
                        {
                            "corpus_id": self.corpus_ids[doc_idx],
                            "score": float(all_scores[query_idx][doc_idx])
                        }
                        for doc_idx in query_sorted_indices
                    ]


                    toolretriever_queries_result_list[query_idx] = sorted(toolretriever_queries_result_list[query_idx], key=lambda x: x["score"])
                    bm25_queries_result_list[query_idx] = sorted(bm25_queries_result_list[query_idx], key=lambda x: x["score"])

       
                    for idx, res in enumerate(toolretriever_queries_result_list[query_idx]):
                        res["score"] = idx
                    for idx, res in enumerate(bm25_queries_result_list[query_idx]):
                        res["score"] = idx


                    toolretriever_queries_result_list[query_idx] = sorted(toolretriever_queries_result_list[query_idx], key=lambda x: x["corpus_id"])
                    bm25_queries_result_list[query_idx] = sorted(bm25_queries_result_list[query_idx], key=lambda x: x["corpus_id"])
     
                    for res_item_tr in toolretriever_queries_result_list[query_idx]:
                        for res_item_bm25 in bm25_queries_result_list[query_idx]:
                            if res_item_tr["corpus_id"] == res_item_bm25["corpus_id"]:
                                queries_result_list[query_idx].append({
                                "corpus_id": res_item_tr["corpus_id"],
                                "score": res_item_tr["score"] + res_item_bm25["score"]
                                })


        elif self.retriever_type == 'reranker':
  
            with torch.no_grad():
                all_scores = self.score_function(self.query_embeddings, self.corpus_embeddings)
                all_scores = all_scores.cpu().numpy()
                sorted_indices = np.argsort(-all_scores, axis=1)
                for query_idx, query in tqdm(enumerate(self.queries_list)):
                    query_sorted_indices = sorted_indices[query_idx] 
                    queries_result_list[query_idx] = [
                        {
                            "corpus_id": self.corpus_ids[doc_idx],
                            "score": float(all_scores[query_idx][doc_idx])
                        }
                        for doc_idx in query_sorted_indices
                    ]
                   
                    queries_result_list[query_idx] = sorted(queries_result_list[query_idx], key=lambda x: x["score"],reverse=True)

                    corpus_id_for_rerank = [res["corpus_id"] for res in queries_result_list[query_idx][:20]]
                    corpus_for_rerank = [self.cid_corpus_dict[id] for id in corpus_id_for_rerank]
                    
                    compare_list = []
                    for corpus in corpus_for_rerank:
                        compare_list.append([query, corpus])
                    reranker_scores = self.reranker.compute_score(compare_list, normalize=True)

                    for idx, res in enumerate(queries_result_list[query_idx]):
                        if idx < 20:
                            res["score"] = reranker_scores[idx]
                        else:
                            res["score"] = -100


       
        
    

        scores_list = self.compute_mertrics(queries_result_list)
        if self.write_csv:
            ndcg = self.save_outcome(scores_list,epoch,steps)
        
        return scores_list, queries_result_list    
        
    def compute_mertrics(self,queries_result_list):    
        scores_list = []
        for query_idx in tqdm(range(len(queries_result_list))):
            # print(query_idx)
            query_id = self.queries_id[query_idx]
            # print(query_id)
            top_hits = queries_result_list[query_idx] 
            sig_dict = self.compute_ndcg_for_query((query_idx,int(query_id),top_hits))
            scores_list.append(sig_dict)
        return scores_list
        
            

    def compute_ndcg_for_query(self,query_tuple):
        k_list = [5, 10]
        sig_dict = {}     
        q_idx, query_id, top_hits = query_tuple
        query_relevant_docs = self.relevant_docs[query_id]
        true_relevance = np.zeros(len(self.corpus_ids))
        predicted_scores = np.zeros(len(self.corpus_ids))
        for hit in top_hits:
            predicted_scores[self.cid_to_idx[int(hit["corpus_id"])]] = hit["score"]
            if int(hit["corpus_id"]) in query_relevant_docs or hit["corpus_id"] in query_relevant_docs:
                true_relevance[self.cid_to_idx[int(hit["corpus_id"])]] = 1
        sig_dict['id'] = query_id
        sig_dict['query'] = self.queries_list[q_idx]
        for k in k_list:
            sig_dict[f'ndcg{k}'] = ndcg_score([true_relevance], [predicted_scores], k=k)
        sig_dict['ndcg'] = np.mean(sig_dict['ndcg5'] + sig_dict['ndcg10'])
        return sig_dict
    
    
    def save_outcome(self,scores_list,epoch,steps):
        ndcg5,ndcg10 = [],[]
        for score in scores_list:
            ndcg5.append(score['ndcg5'])
            ndcg10.append(score['ndcg10'])
        avg_ndcg = [sum(ndcg5)/len(ndcg5),sum(ndcg10)/len(ndcg10)]
        json_file = os.path.join(self.output_path, f"{self.phase}_Retrieval_evaluation_results.json")
        with open(json_file, 'w') as f:
            average_ndcg = sum(avg_ndcg) / len(avg_ndcg)
            json_data = {"average_ndcg": average_ndcg}
            json.dump(json_data,f, indent=4)
        csv_path = os.path.join(self.output_path, f"{self.phase}_Retrieval_evaluation_results.csv")
        csv_headers = [
            "epoch",
            "steps",
            "Average NDCG@5",
            "Average NDCG@10"
        ]
        if not os.path.isfile(csv_path):
            fOut = open(csv_path, mode="w", encoding="utf-8")
            fOut.write(",".join(csv_headers))
            fOut.write("\n")
        else:
            fOut = open(csv_path, mode="a", encoding="utf-8")
        output_data = [epoch, steps]
        output_data.append(avg_ndcg)
        fOut.write(",".join(map(str, output_data)))
        fOut.write("\n")
        fOut.close()
        return avg_ndcg