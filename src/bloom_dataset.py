import torch
import json
import torch.distributed as dist
from torch.utils.data import IterableDataset, 
from transformers import AutoModelForCausalLM, AutoTokenizer

class DistributedSampler:
    """Get data based on work_id.
    """
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data





class DataList():
    def __init__(self, file_list, tokenizer, max_length = 2048, window_size=20, train_on_inputs=False):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.window_size = window_size
        self.max_length = max_length

    def __iter__(self):
        with open(self.file_list, "r", encoding="utf-8") as file:
            for sub_file in file.readlines() :
                json_file = open(sub_file.strip(), "r", encoding="utf-8")
                json_list = json.loads(json_file.read())
                for line in json_list:
                    data = line["title"]+":"+line["content"]
                    full_prompt = self.tokenizer.bos_token + data
                    full_data = self.tokenizer(full_prompt, return_tensors=None)
                    i = 0
                    while i < len(full_data["input_ids"]):
                        input_ids = full_data["input_ids"][i:i+self.max_length].copy()
                        attention_mask =  full_data["attention_mask"][i:i+self.max_length].copy()
                        new_result = {"input_ids":input_ids, "attention_mask":attention_mask}
                        result = self.tokenize(new_result)
                        i = i+self.max_length
                        yield result
                

    def tokenize(self, result, add_eos_token=True):
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= self.max_length:
            result["input_ids"][self.max_length - 1] = self.tokenizer.eos_token_id
            result["attention_mask"][self.max_length - 1] = 1

        result["labels"] = result["input_ids"].copy()
        return result




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bloom-3b")
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)
    print(tokenizer.bos_token_id)
    print(tokenizer.bos_token)
    dataset = DataList("wudao.lst", tokenizer)
    for id, data in enumerate(dataset):
        # print(data)
        print(len(data["input_ids"]))
        # if id == 2:
        #     exit(0)