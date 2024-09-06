import matplotlib.pyplot as plt
import time
import torch
import heapq
from collections import defaultdict
from typing import List, Optional, Callable
import random
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class ReqToTokenPool:
    def __init__(self):
        self.req_to_token = {}

    def add_request(self, req_idx, tokens):
        self.req_to_token[req_idx] = tokens

    def free(self, req_idx):
        if req_idx in self.req_to_token:
            del self.req_to_token[req_idx]


class BaseTokenToKVPool:
    def __init__(self, size: int):
        self.size = size
        self.kv_store = {}

    def free(self, kv_indices):
        for idx in kv_indices:
            if idx in self.kv_store:
                del self.kv_store[idx]

    def add_kv(self, token_idx, value):
        if len(self.kv_store) < self.size:
            self.kv_store[token_idx] = value
        else:
            print("KV Pool is full!")


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache:
    def __init__(
            self,
            req_to_token_pool: ReqToTokenPool,
            token_to_kv_pool: BaseTokenToKVPool,
            disable: bool = False,
            log_evictions: bool = False,
            eviction_log_file: str = "eviction_log.txt"
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.log_evictions = log_evictions
        self.eviction_log_file = eviction_log_file
        self.reset()

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)

        if value:
            try:
                if all(isinstance(v, torch.Tensor) for v in value):
                    value = torch.concat(value)
                else:
                    value = value
            except ValueError as e:
                print(f"Error converting to tensor: {e}")
                value = value
        else:
            value = torch.tensor([], dtype=torch.int32)

        return value, last_node[0]

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
                     req.req_pool_idx, : len(token_ids)
                     ]

        if self.disable:
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices): new_prefix_len])
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
                     req.req_pool_idx, : len(token_ids)
                     ]

        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices): new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids)
        assert len(new_indices) == len(token_ids)
        self.req_to_token_pool.req_to_token[
        req.req_pool_idx, len(req.prefix_indices): len(new_indices)
        ] = new_indices[len(req.prefix_indices):]

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue
            self.log_eviction(x)
            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def evict_with_lookahead(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        eviction_schedule = self.load_eviction_log()
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            key_str = str(x.key)
            if key_str in eviction_schedule:
                future_eviction_time = eviction_schedule[key_str]
                # Prioritize evictions for nodes that are scheduled farthest in the future
                if future_eviction_time > time.time() + 300:
                    evict_callback(x.value)
                    num_evicted += len(x.value)
                    self._delete_leaf(x)  #keep something that will be reused in the future, otherwise we should immediately delete it.

                    if len(x.parent.children) == 0:
                        heapq.heappush(leaves, x.parent)

    def load_eviction_log(self):
        eviction_schedule = {}
        with open(self.eviction_log_file, 'r') as log_file:
            for line in log_file:
                parts = line.strip().split(", ")
                key = parts[0].split(": ")[1]
                evict_time = float(parts[1].split(": ")[1])
                eviction_schedule[key] = evict_time
        return eviction_schedule

    def log_eviction(self, node: TreeNode):
        if self.log_evictions:
            with open(self.eviction_log_file, 'a') as log_file:
                log_file.write(f"Evicting node with key: {node.key}, time: {time.time()}\n")

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
            self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


def run_cache_test(cache, arrival_rate, prompt_length, generation_length):
    hit_count = 0
    total_requests = 0
    time_series = []
    hit_rate_series = []
    process_time_series = []

    # 生成部分重复的请求集
    base_sequence = ''.join(random.choices('ABCDE12345!@#$%', k=prompt_length // 2))

    frequent_requests = [
        base_sequence + ''.join(random.choices('ABCDE12345!@#$%', k=prompt_length // 2))
        for _ in range(3)
    ]

    infrequent_requests = [
        ''.join(random.choices('FGHIJ67890^&*()', k=generation_length))
        for _ in range(30)  # Reduce the number of infrequent requests
    ]

    # 构建部分重复的请求集
    partially_repeating_requests = []
    for _ in range(300):  # Increase the number of partially repeating requests
        start = random.randint(0, len(base_sequence) - 5)
        partially_repeating_part = base_sequence[start:start + 5]
        remaining_length = prompt_length - len(partially_repeating_part)
        new_request = partially_repeating_part + ''.join(random.choices('ABCDE12345!@#$%', k=remaining_length))
        partially_repeating_requests.append(new_request)

    request_set = frequent_requests * 10 + partially_repeating_requests + infrequent_requests
    random.shuffle(request_set)

    for req in request_set:
        start_time = time.time()  # 开始计时

        total_requests += 1
        _, node = cache.match_prefix(req)

        if node and node.value:
            hit_count += 1

        cache.insert(req)

        end_time = time.time()  # 结束计时
        process_time = end_time - start_time
        process_time_series.append(process_time)

        # 模拟请求的到达率，控制请求的频率
        time.sleep(1 / arrival_rate)

        if total_requests % 10 == 0:
            hit_rate = hit_count / total_requests
            avg_process_time = sum(process_time_series) / len(process_time_series)
            time_series.append(total_requests)
            hit_rate_series.append(hit_rate)
            process_time_series = []

    return {
        'time': time_series,
        'hit_rate': hit_rate_series,
        'avg_process_time': avg_process_time
    }


if __name__ == "__main__":
    random.seed(42)

    arrival_rate = 2000  # Reduce the arrival rate to simulate a more controlled environment
    prompt_length = 512  # Reduce the prompt length to increase the likelihood of repetitions
    generation_length = 1024  # Reduce the generation length to make infrequent requests less significant

    req_to_token_pool = ReqToTokenPool()
    token_to_kv_pool = BaseTokenToKVPool(size=500)  # Reduce KV pool size to make cache capacity more constrained

    for i in range(500):
        token_to_kv_pool.add_kv(i, torch.tensor([i], dtype=torch.int32))

    for i in range(10):
        req_to_token_pool.add_request(i, [random.randint(0, 499) for _ in range(5)])

    sglang_cache = RadixCache(req_to_token_pool, token_to_kv_pool, disable=False)
    sglang_results = run_cache_test(sglang_cache, arrival_rate, prompt_length, generation_length)

    lookahead_cache = RadixCache(req_to_token_pool, token_to_kv_pool, disable=False, log_evictions=True,
                                 eviction_log_file="eviction_log.txt")
    lookahead_results_first_run = run_cache_test(lookahead_cache, arrival_rate, prompt_length, generation_length)
    lookahead_cache = RadixCache(req_to_token_pool, token_to_kv_pool, disable=False, log_evictions=False,
                                 eviction_log_file="eviction_log.txt")
    lookahead_results_second_run = run_cache_test(lookahead_cache, arrival_rate, prompt_length, generation_length)

    lookahead_hit_rates = np.array(lookahead_results_second_run['hit_rate'])
    sglang_hit_rates = np.array(sglang_results['hit_rate'])

    plt.figure(figsize=(10, 6))
    plt.plot(sglang_results['time'], sglang_results['hit_rate'], label='SGLang - Hit Rate', marker='o')
    plt.plot(lookahead_results_second_run['time'], lookahead_results_second_run['hit_rate'],
             label='Lookahead - Hit Rate', marker='x')
    plt.xlabel('Number of Requests')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Hit Rate Comparison: SGLang vs Lookahead')
    plt.legend()
    plt.show()

