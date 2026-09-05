import os
import json
import csv
import shutil
import argparse

class BFCLProcessor:
    def __init__(self, data_calling_dir, possible_answer_dir, output_dir):
        self.data_calling_dir = data_calling_dir
        self.possible_answer_dir = possible_answer_dir
        self.output_dir = output_dir
        self.qid_question_dict = {}
        self.name_to_docid = {}
        self.corpus_data = []

    def build_corpus(self):
        print("Step 1: Building global corpus...")
        all_fun_str_dict = {}
        
        for subset in os.listdir(self.data_calling_dir):
            if "BFCL_" in subset:
                data_path = os.path.join(self.data_calling_dir, subset)
                if not os.path.isfile(data_path):
                    continue
                
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data_item = json.loads(line)
                        question = ""
                        for message in data_item.get("question", []):
                            for conv in message:
                                question += " " + conv["content"].replace("\n", "").strip()
                        
                        self.qid_question_dict[data_item["id"]] = question.strip()

                        for fun in data_item.get("function", []):
                            all_fun_str_dict[fun["name"]] = json.dumps(fun)

        for idx, (name, fun_str) in enumerate(all_fun_str_dict.items()):
            self.corpus_data.append({"docid": idx, "document_content": fun_str})
            self.name_to_docid[name] = idx

        os.makedirs(self.output_dir, exist_ok=True)
        temp_corpus_path = os.path.join(self.output_dir, "corpus.tsv")
        self._write_tsv(temp_corpus_path, self.corpus_data)
        return temp_corpus_path

    def process_subsets(self, categories, temp_corpus_path):
        print(f"Step 2: Processing {len(categories)} categories...")
        
        for cat in categories:
            subset_name = f"BFCL_v4_{cat}"
            ans_path = os.path.join(self.possible_answer_dir, f"{subset_name}.json")
            
            if not os.path.exists(ans_path):
                print(f"Warning: {ans_path} not found. Skipping.")
                continue

            subset_dir = os.path.join(self.output_dir, subset_name, "ori")
            os.makedirs(subset_dir, exist_ok=True)
            shutil.copy(temp_corpus_path, os.path.join(subset_dir, "corpus.tsv"))

            queries = []
            qrels = []

            with open(ans_path, "r", encoding="utf-8") as f:
                for q_idx, line in enumerate(f):
                    data_item = json.loads(line)
                    qid_str = data_item["id"]
                    
                    queries.append({
                        "qid": q_idx,
                        "question": self.qid_question_dict.get(qid_str, "")
                    })

                    gt_funs = data_item.get("ground_truth", [])
                    for gt_item in gt_funs:
                        gt_name = list(gt_item.keys())[0]
                        docid = self._find_docid(gt_name)
                        if docid is not None:
                            qrels.append({"qid": q_idx, "head": 0, "fid": docid, "tail": 1})

            self._write_tsv(os.path.join(subset_dir, "test.query.txt"), queries)
            self._write_tsv(os.path.join(subset_dir, "qrels.test.tsv"), qrels)

        os.remove(temp_corpus_path)
        print("All tasks completed.")

    def _find_docid(self, target_name):
        if target_name in self.name_to_docid:
            return self.name_to_docid[target_name]
        for name, docid in self.name_to_docid.items():
            if target_name == name.split(".")[-1]:
                return docid
        return None

    def _write_tsv(self, file_path, data_list):
        if not data_list: return
        fieldnames = data_list[0].keys()
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if "docid" in fieldnames:
                writer.writeheader()
            writer.writerows(data_list)

def main():
    parser = argparse.ArgumentParser(description="BFCL Dataset Preprocessor for Tool Retrieval")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to bfcl_data_standard")
    parser.add_argument("--ans_dir", type=str, required=True, help="Path to possible_answer folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Target path for output")
    parser.add_argument("--categories", type=str, 
                        default="multiple,parallel_multiple,parallel,simple_java,simple_javascript,simple_python,live_multiple,live_parallel,live_simple,live_parallel_multiple",
                        help="Comma separated list of test categories")

    args = parser.parse_args()
    cat_list = args.categories.split(",")

    processor = BFCLProcessor(args.input_dir, args.ans_dir, args.output_dir)
    corpus_path = processor.build_corpus()
    processor.process_subsets(cat_list, corpus_path)

if __name__ == "__main__":
    main()