import os
import json
import csv
import shutil
import argparse

class AceBenchProcessor:
    def __init__(self, data_calling_dir, possible_answer_dir, output_dir):
        self.data_calling_dir = data_calling_dir
        self.possible_answer_dir = possible_answer_dir
        self.output_dir = output_dir
        self.qid_question_dict = {}
        self.name_to_docid = {}
        self.corpus_data = []

    def build_corpus(self):
        print("Step 1: Scanning original data to build global corpus...")
        all_fun_str_dict = {}
        
        if not os.path.exists(self.data_calling_dir):
            raise FileNotFoundError(f"Source directory {self.data_calling_dir} not found.")

        for subset in os.listdir(self.data_calling_dir):
            if "normal_" in subset:
                data_path = os.path.join(self.data_calling_dir, subset)
                if not os.path.isfile(data_path):
                    continue
                
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        data_item = json.loads(line)
                        
                        question = data_item.get("question", "").replace("\n", "").strip()
                        self.qid_question_dict[data_item["id"]] = question

                        for fun in data_item.get("function", []):
                            all_fun_str_dict[fun["name"]] = json.dumps(fun)

        for idx, (name, fun_str) in enumerate(all_fun_str_dict.items()):
            self.corpus_data.append({"docid": idx, "document_content": fun_str})
            self.name_to_docid[name] = idx

        os.makedirs(self.output_dir, exist_ok=True)
        temp_corpus_path = os.path.join(self.output_dir, "corpus.tsv")
        self._write_tsv(temp_corpus_path, self.corpus_data, write_header=True)
        return temp_corpus_path

    def process_subsets(self, categories, temp_corpus_path):
        print(f"Step 2: Processing {len(categories)} categories...")
        
        for cat in categories:
            subset_filename = f"data_{cat}"
            ans_path = os.path.join(self.possible_answer_dir, f"{subset_filename}.json")
            
            if not os.path.exists(ans_path):
                print(f"Warning: Answer file {ans_path} not found. Skipping.")
                continue

            subset_dir = os.path.join(self.output_dir, subset_filename, "ori")
            os.makedirs(subset_dir, exist_ok=True)
            shutil.copy(temp_corpus_path, os.path.join(subset_dir, "corpus.tsv"))

            query_items = []
            qrel_items = []

            with open(ans_path, "r", encoding="utf-8") as f:
                for q_idx, line in enumerate(f):
                    if not line.strip(): continue
                    data_item = json.loads(line)
                    qid_in_file = data_item["id"]
                    
                    query_items.append({
                        "qid": q_idx,
                        "question": self.qid_question_dict.get(qid_in_file, "")
                    })

                    gt_list = data_item.get("ground_truth", [])
                    if not isinstance(gt_list, list):
                        gt_list = [gt_list]

                    for gt_obj in gt_list:
                        gt_fun_name = list(gt_obj.keys())[0]
                        fid = self._find_fid_with_ace_logic(gt_fun_name)
                        if fid is not None:
                            qrel_items.append({
                                "qid": q_idx,
                                "head": 0,
                                "fid": fid,
                                "tail": 1
                            })

            self._write_tsv(os.path.join(subset_dir, "test.query.txt"), query_items)
            self._write_tsv(os.path.join(subset_dir, "qrels.test.tsv"), qrel_items)

        if os.path.exists(temp_corpus_path):
            os.remove(temp_corpus_path)
        print("Done!")

    def _find_fid_with_ace_logic(self, gt_name):
        for name, fid in self.name_to_docid.items():
            name_suffix = name.split(".")[-1]
            gt_prefix = gt_name.split("_")[0]
            
            if gt_name == name or \
               gt_name == name_suffix or \
               gt_prefix == name or \
               gt_prefix == name_suffix:
                return fid
        return None

    def _write_tsv(self, path, data_list, write_header=False):
        if not data_list: return
        fieldnames = data_list[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerows(data_list)

def main():
    parser = argparse.ArgumentParser(description="AceBench Data Preprocessor")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to data_en (contains normal_*.json)")
    parser.add_argument("--ans_dir", type=str, required=True, help="Path to possible_answer folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output retrieval data path")
    parser.add_argument("--categories", type=str, 
                        default="normal_single_turn_single_function,normal_single_turn_parallel_function,normal_multi_turn_user_adjust,normal_multi_turn_user_switch,normal_similar_api,normal_preference,normal_atom_bool,normal_atom_enum,normal_atom_number,normal_atom_list,normal_atom_object_deep,normal_atom_object_short",
                        help="Comma separated categories")

    args = parser.parse_args()
    cats = [c.strip() for c in args.categories.split(",")]

    processor = AceBenchProcessor(args.input_dir, args.ans_dir, args.output_dir)
    temp_corpus = processor.build_corpus()
    processor.process_subsets(cats, temp_corpus)

if __name__ == "__main__":
    main()