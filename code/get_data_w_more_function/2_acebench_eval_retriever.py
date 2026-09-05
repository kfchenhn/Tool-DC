import argparse
import os
import json
import shutil
from src.utils import initialize_retriever, read_query
from src.api_evaluator import APIEvaluator

class BatchRetrieverEvaluator:
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"Using GPU: {args.gpu_id}")

    def run(self):
        if not os.path.exists(self.args.data_root):
            print(f"Error: Data root {self.args.data_root} does not exist.")
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

                self.process_single_dataset(data_dir)

    def process_single_dataset(self, data_dir):
        print(f"\n>>> Processing: {data_dir}")
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

        self.save_top_k_results(queries_result_list, output_dir)

        score_save_path = os.path.join(output_dir, f'{self.args.phase}_score.json')
        with open(score_save_path, 'w', encoding='utf-8') as f:
            json.dump(score_list, f, indent=4)
        print(f"Score saved to: {score_save_path}")

    def save_top_k_results(self, queries_result_list, output_dir):
        result_output = []
        for q_idx, corpus_id_w_score_list in enumerate(queries_result_list):
            sorted_list = sorted(corpus_id_w_score_list, key=lambda x: x["score"], reverse=True)
            top_k_ids = [item["corpus_id"] for item in sorted_list][:self.args.top_k]

            result_output.append({
                "qid": q_idx,
                "corpus_id_list": top_k_ids
            })

        save_path = os.path.join(output_dir, f'{self.args.phase}_retr_corpus_id.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_output, f, indent=4)
        print(f"Top-{self.args.top_k} results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for Tool Retrieval")
    
    parser.add_argument("--model_path", default="/models/ToolBench_IR_bert_based_uncased", type=str)
    parser.add_argument("--data_root", required=True, type=str, help="Root dir containing subset/ori/ structure")
    
    parser.add_argument("--retriever_type", default='ToolRetriever', choices=['bm25', 'TF-IDF', 'ada', 'ToolRetriever', 'colbert'], type=str)
    parser.add_argument("--phase", default="test", type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--top_k", default=50, type=int, help="Number of documents to retrieve")
    
    parser.add_argument('--gpu_id', type=str, default="0")
    
    args = parser.parse_args()

    evaluator = BatchRetrieverEvaluator(args)
    evaluator.run()

if __name__ == '__main__':
    main()