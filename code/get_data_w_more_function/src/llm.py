import requests
import time
import random
from datetime import datetime
from tqdm import tqdm

fuzzy_template = """
You are given a user query and a list of related APIs. Your task is to generate a fuzzier version of the user query by simplifying or replacing technical terms with synonyms, without changing the user's core requirements. You can ollow these steps:
- Analyze the user query: Identify how many tasks the user has and what the specific needs are.
- Compare with the API list: Match the user's tasks with the relevant APIs. Remove any references to specific APIs or redundant technical details.
- Simplify technical terms: Replace highly specialized or technical terms with more common, everyday language. The goal is to make the query sound like something a regular user would say in casual conversation.
- Rephrase the query: Use simpler language or synonyms where appropriate, but ensure the core intent remains unchanged.
- Output the result: Provide the final fuzzier version of the query.

Example1:
Original Query: I'm organizing a gaming tournament for my company's employees. Could you provide the statistics and ratings of highly skilled players in popular games like Dota 2 and World of Tanks? Also, recommend some gaming peripherals and accessories for the event.
Relevant APIs: [tool_name:World of Tanks Stats,api_name:Get Stats],[tool_name:DOTA 2 Steam Web,api_name:Match History],[tool_name:DOTA 2 Steam Web,api_name:Match Details],[tool_name:CheapShark - Game Deals,api_name:List of Deals],[tool_name:CheapShark - Game Deals,api_name:Game Lookup]
Answer:I'm organizing a company gaming tournament and need player stats for top players in popular games. Can you also recommend some good gaming gear for the event?

Example2: 
Original Query: I want to surprise my family with a personalized playlist. Can you recommend some popular tracks from different genres? Additionally, provide me with the detailed information of a playlist that I want to create. Also, fetch the track URL of a specific song that I want to include in the playlist.
Relevant APIs: [tool_name:Shazam,api_name:artists/get-summary],[tool_name:Deezer,api_name:Track],[tool_name:Soundcloud,api_name:/playlist/info]
Answer: I want to make a special playlist for my family. Can you suggest some hit songs from different music styles? Also, give me more info about the playlist I'm putting together. Finally, can you get me the link to a specific track I want to add?

Now, Please make the fuzzier query. 
Original Query: {instruction}
Relevant APIs: {apis}
Answer:
"""

class ChatGpt:
    def __init__(self, model_name, api_key="", api_url="", rate_limit=None):
        self.model = model_name
        self.api_key = api_key
        self.api_url = api_url

        self.temperature = 0.0
        self.top_p = 0.95
        self.top_k = 40
        self.max_output_tokens = 1024
        self.candidate_count = 1
        self.repetition_penalty = 1
        self.stream = False


    def retry_request(self, func, *args, max_retries=5, retry_delay=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay + random.uniform(0, 1))
                else:
                 
                    raise

    def raw_request(self, model, messages, temperature=0.0, timeout=10):
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        return response.json()

    def get_query(self, prompt_text):

        messages = [{"role": "user", "content": prompt_text}]
        return self.retry_request(self._get_chat_completion, messages)

    def _get_chat_completion(self, messages):
        response = self.raw_request(self.model, messages, self.temperature)
        return response["choices"][0]["message"]["content"]

    

    def get_embeddings(self, text, max_retry=100):
  
        # for retry in range(max_retry):
        
        for retry in range(max_retry):
            try:
                url = f"{self.api_url}/embeddings"
                payload = {"model": self.model, "input": text}
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, headers=headers, json=payload)
                emb_content = response.json()["data"][0]["embedding"]
                break
            except:
                time.sleep(10)
        return emb_content
        
                
    def get_score(self, instruction):
        prompt = fuzzy_template.format(instruction=instruction, apis="[]")
        return self.get_query(prompt)

    def fuzzy_generate(self, batch_text, batch_id, id_text_dict):
        for text, qid in zip(batch_text, batch_id):
            response = self.get_query(text)
            id_text_dict[qid] = response

    def dataset_generate(self, batch_text, text_emb_dict):
        for text in batch_text:
            embedding = self.get_embeddings(text)
            text_emb_dict[text] = embedding
    
    def dataset_valid(self, batch_data):
        results = []
        for item in batch_data:
            id_, text = next(iter(item.items()))
            score = self.get_score(text)
            results.append({id_: score})
            time.sleep(1 / self.rate_limit.get("RPM", 100))
        return results
