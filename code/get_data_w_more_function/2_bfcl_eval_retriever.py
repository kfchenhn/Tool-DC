import argparse
import os
import json
import shutil
from src.utils import initialize_retriever, read_query
from src.api_evaluator import APIEvaluator

class BFCLEvaluator:
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_id}")

    def run_batch_evaluation(self):
        if not os.path.exists(self.args.data_root):
            print(f"Error: Data root {self.args.data_root} not found.")
            return

        for sub_data_name in os.listdir(self.args.data_root):
            sub_path = os.path.join(self.args.data_root, sub_data_name)
            if not os.path.isdir(sub_path):
                continue

            for sub2_data_name in os.listdir(sub_path):
                data_dir = os.path.join(sub_path, sub2_data_name)
                if not os.path.isdir(data_dir):
                    continue
                
                query_file = os.path.join(data_dir, f"{self.args.phase}.query.txt")
                if not os.path.exists(query_file):
                    continue

                self._process_subset(data_dir)

    def _process_subset(self, data_dir):
        print(f"\n[Starting] Processing directory: {data_dir}")
        
        output_dir = os.path.join(data_dir, self.args.retriever_type)
        os.makedirs(output_dir, exist_ok=True)

        ir_corpus, ir_relevant_docs, id_corpus_emb_dict, retriever_model, bm25, tf_idf = initialize_retriever(
            self.args.model_path,
            retriever_type=self.args.retriever_type,
            output_dir=data_dir,
            phase=self.args.phase,
        )

        query_path = os.path.join(data_dir, f"{self.args.phase}.query.txt")
        qid_list, query_list = read_query(query_path)

        ir_evaluator = APIEvaluator(
            query_list, qid_list, ir_corpus, ir_relevant_docs,
            id_corpus_emb_dict, retriever_model, bm25, tf_idf,
            self.args.retriever_type, output_dir,
            write_csv=True, phase=self.args.phase
        )
        
        score_list, queries_result_list = ir_evaluator()

        self._save_retrieval_results(queries_result_list, output_dir)

        score_file = os.path.join(output_dir, f'{self.args.phase}_score.json')
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(score_list, f, indent=4)
        
        print(f"[Finished] Results saved in: {output_dir}")

    def _save_retrieval_results(self, queries_result_list, output_dir):
        q_and_retrieved_corpus_id_list = []
        
        for qid, corpus_id_w_score_list in enumerate(queries_result_list):
            sorted_results = sorted(corpus_id_w_score_list, key=lambda x: x["score"], reverse=True)
            top_k_ids = [item["corpus_id"] for item in sorted_results][:self.args.top_k]

            q_and_retrieved_corpus_id_list.append({
                "qid": qid,
                "corpus_id_list": top_k_ids
            })

        id_results_file = os.path.join(output_dir, f'{self.args.phase}_retr_corpus_id.json')
        with open(id_results_file, 'w', encoding='utf-8') as f:
            json.dump(q_and_retrieved_corpus_id_list, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for BFCL Retrieval")
    
    parser.add_argument("--data_root", default="", 
                        type=str, help="Root directory for retrieve data")
    parser.add_argument("--model_path", default="", 
                        type=str, help="Path to the pre-trained/fine-tuned model")
    
    parser.add_argument("--retriever_type", default='ToolRetriever', 
                        choices=['bm25', 'TF-IDF', 'ada', 'ToolRetriever', 'colbert'], help="Retriever type")
    parser.add_argument("--phase", default="test", type=str, help="test or train phase")
    parser.add_argument("--top_k", default=50, type=int, help="Number of items to retrieve per query")
    parser.add_argument("--max_length", default=256, type=int, help="Max sequence length")
    
    parser.add_argument('--gpu_id', type=str, default="1", help="GPU device ID")

    args = parser.parse_args()

    evaluator = BFCLEvaluator(args)
    evaluator.run_batch_evaluation()

if __name__ == '__main__':
    main()