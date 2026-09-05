from abc import ABC, abstractmethod
import requests
import json
from datetime import datetime
import time
import tqdm
import threading
import datetime
from datetime import datetime
import random
import requests
from tqdm import tqdm
import openai
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

Now, Please make the fuzzier query. Only the content after "Answer" needs to be output, and no other words need to be output.
Original Query: {instruction}
Relevant APIs: {apis}
Answer:
"""
class BaseModel(ABC):
    def __init__(self, rateLimit={}, ) -> None:
        """
        rateLimit setting
        """
        super().__init__()

        self.thread_pools = []

        self.rateLimit = rateLimit

        if "RPM" in self.rateLimit:
            # RPM : the request rate limit per minute
            # init time for Rate Limit
            self._minute = datetime.now().minute
            self._avail_req = self.rateLimit["RPM"]
        elif "SPR" in self.rateLimit:
            # SPR : second per request (low speed request)
            self._lastTime = datetime.now()


    def configure_params(self, temperature=1.0, top_p=0.95, top_k=40, max_output_tokens=1024, candidate_count=1, repetition_penalty=1, stream=False):
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.max_output_tokens=max_output_tokens
        self.candidate_count = candidate_count
        self.repetition_penalty = repetition_penalty
        self.stream = stream


    def isRateLimited(self)->bool:
        if "RPM" in self.rateLimit:
            return self._isRateLimited_RPM()
        elif "SPR" in self.rateLimit:
            return self._isRateLimited_SPR()
        else:
            print("please check! Not rate limit found!")
            return False


    def _isRateLimited_RPM(self)->bool:
        cur_time = datetime.now().minute
        if cur_time != self._minute:
            # time diff, reset avail request
            self._avail_req = self.rateLimit["RPM"]
            self._minute = cur_time

        if self._avail_req > 0:
            self._avail_req -= 1
            return False
        else:
            return True


    def _isRateLimited_SPR(self)->bool:
        cur_time = datetime.now()
        diff = (cur_time-self._lastTime).total_seconds()
        if diff >= (self.rateLimit["SPR"]+0.1):
            self._lastTime = cur_time
            return False
        else:
            return True        
    def fuzzy_generate(self,batch_text,batch_id,id_text_dict):
        def my_function(text,qid,id_text_dict,lock):
            max_retries = 50
            retry_delay = 5

            for i in range(max_retries):
                try:
                    fuzzy_query = self.get_query(text)
                    with lock:
                        id_text_dict[qid] = fuzzy_query
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        time.sleep(retry_delay + random.uniform(0, 1))
                    else:
                        tqdm.write(f"获取 {text} 的失败,已达到最大重试次数")
                        raise
        lock = threading.Lock()
        for text,qid in zip(batch_text,batch_id):
            while True:
                lock.acquire()
                rateLimit = self.isRateLimited()
                lock.release()
                if rateLimit:
                    time.sleep(1)
                else:
                    break  
            thread = threading.Thread(target=my_function,args=(text,qid,id_text_dict, lock))
            self.thread_pools.append(thread)
            thread.start()
            try:
                sleep_time =  1 / self.rateLimit["RPM"]
            except:
                sleep_time = 0
            time.sleep(sleep_time)        
        for thread in self.thread_pools:
            thread.join()       
        self.thread_pools.clear()
    
    def dataset_valid(self,batch_data):
        batch_qid_query_dict = []
        def valid_function(id,text,lock):
            max_retries = 50
            retry_delay = 10
            for i in range(max_retries):
                try:
                    response = self.get_score(text)
                    batch_qid_query_dict.append({id:response})
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    if i < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        pass


        lock = threading.Lock() 
        # print(batch_data)  
        for item in batch_data: 
            # print(item)
            id, text = next(iter(item.items()))
            while True:
                lock.acquire()
                rateLimit = self.isRateLimited()
                lock.release()
                if rateLimit:
                    time.sleep(1)
                else:
                    break  
            thread = threading.Thread(target=valid_function,args=(id,text, lock))
            self.thread_pools.append(thread)
            thread.start()
            try:
                sleep_time =  1 / self.rateLimit["RPM"]
            except:
                sleep_time = 0
            time.sleep(sleep_time)        
        for thread in self.thread_pools:
            thread.join()       
        self.thread_pools.clear()
        return batch_qid_query_dict

    def dataset_generate(self, batch_text,text_emb_dict):
        def my_new_function(text,text_emb_dict,lock):
            max_retries = 50
            retry_delay = 5
            for i in range(max_retries):
                try:
                    emb = self.get_embeddings(text)
                    with lock:
                        text_emb_dict[text] = emb
                    break 
                except:
                    if i < max_retries - 1:
                       
                        time.sleep(retry_delay + random.uniform(0, 1))
                    else:
                        tqdm.write(f"获取 {text} 的嵌入失败,已达到最大重试次数")
                    continue
        lock = threading.Lock()       
        # call api
        for text in batch_text:    
            while True:
                lock.acquire()
                rateLimit = self.isRateLimited()
                lock.release()
                if rateLimit:
                    time.sleep(1)
                else:
                    break  
            thread = threading.Thread(target=my_new_function,args=(text,text_emb_dict, lock))
            self.thread_pools.append(thread)
            thread.start()
            try:
                sleep_time =  1 / self.rateLimit["RPM"]
            except:
                sleep_time = 0
            time.sleep(sleep_time)        
        for thread in self.thread_pools:
            thread.join()       
        self.thread_pools.clear()


class ChatGpt(BaseModel):
    def __init__(self, model_name, api_key="", api_url="") -> None:
        # gpt-3.5-turbo
        # gpt-4
        rateLimit={
            "RPM":200

            }

        super().__init__(rateLimit)           
        self.model=model_name


        self.api_key = api_key
        self.api_url = api_url

        self.configure_params(temperature=0)    
    def completions_with_backoff(self,**kwargs):
        return openai.ChatCompletion.create(**kwargs)
    def get_embeddings(self, text):
        url = self.api_url + "/embeddings"

        payload = json.dumps({
        "model": self.model,
        "input": text
        })
        headers = {
        'Authorization': f'Bearer {self.api_key}',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        
        return response.json()["data"][0]["embedding"]
    def raw_request(self, model, messages, temperature, timeout=10):
        import requests

        url = self.api_url + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.api_key)
        }

        data = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=data, timeout=timeout) 
        result = response.json()
        return result          
    def get_query(self,text):
        messages = [{"role":"user","content":text}]
        response = self.raw_request(self.model, messages, self.temperature)
        return response["choices"][0]["message"]["content"]
    def get_score(self,instruction):
        text = human_template.format(instruction=instruction)
        messages = [{"role":"user","content":text}]
        response = self.raw_request(self.model, messages, self.temperature)
        return response["choices"][0]["message"]["content"]