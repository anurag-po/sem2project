"""
  1. TRIE (Prefix Tree)  : Low-level prefix indexing, word frequency tracking,
                           and auto-complete suggestions.
  2. STACK (LIFO)        : Boolean query parsing (Shunting-Yard algorithm)
                           and Reverse Polish Notation (RPN) evaluation.
  3. QUEUE (FIFO)        : Document ingestion pipeline and level-order traversals.
  4. MAX-HEAP            : Priority queue for ranked TF-IDF search retrieval (Top-K).
  5. BINARY SEARCH TREE  : Document metadata storage and ordered catalog traversal.
  6. INVERTED INDEX      : Term-to-posting-list index for instant multi-doc lookup.
================================================================================
"""

from __future__ import annotations
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Generator, List, Optional, Set, Tuple


# ==============================================================================
# 1. CORE DATA STRUCTURE: STACK (LIFO)
# ==============================================================================

class StackNode:
    """Node for a linked-list implementation of a Stack."""
    def __init__(self, value: Any, next_node: Optional[StackNode] = None):
        self.value = value
        self.next = next_node


class Stack:
    """
    Linked LIFO (Last-In, First-Out) Stack.
    Time Complexity:
      - Push: O(1)
      - Pop:  O(1)
      - Peek: O(1)
    """
    def __init__(self) -> None:
        self._top: Optional[StackNode] = None
        self._size: int = 0

    def push(self, item: Any) -> None:
        """Pushes an element onto the top of the stack."""
        self._top = StackNode(item, self._top)
        self._size += 1

    def pop(self) -> Any:
        """Removes and returns the top element of the stack."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        assert self._top is not None
        val = self._top.value
        self._top = self._top.next
        self._size -= 1
        return val

    def peek(self) -> Any:
        """Returns the top element without removing it."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        assert self._top is not None
        return self._top.value

    def is_empty(self) -> bool:
        """Returns True if the stack contains no elements."""
        return self._top is None

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        items = []
        curr = self._top
        while curr:
            items.append(repr(curr.value))
            curr = curr.next
        return f"Stack([{', '.join(items)}])"


# ==============================================================================
# 2. CORE DATA STRUCTURE: QUEUE (FIFO)
# ==============================================================================

class QueueNode:
    """Node for a linked-list implementation of a Queue."""
    def __init__(self, value: Any):
        self.value = value
        self.next: Optional[QueueNode] = None


class Queue:
    """
    Linked FIFO (First-In, First-Out) Queue.
    Used for document ingestion queues and breadth-first search traversals.
    Time Complexity:
      - Enqueue: O(1)
      - Dequeue: O(1)
      - Peek:    O(1)
    """
    def __init__(self) -> None:
        self._head: Optional[QueueNode] = None
        self._tail: Optional[QueueNode] = None
        self._size: int = 0

    def enqueue(self, item: Any) -> None:
        """Inserts an element at the rear of the queue."""
        new_node = QueueNode(item)
        if self._tail is not None:
            self._tail.next = new_node
            self._tail = new_node
        else:
            self._head = self._tail = new_node
        self._size += 1

    def dequeue(self) -> Any:
        """Removes and returns the front element from the queue."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        assert self._head is not None
        val = self._head.value
        self._head = self._head.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return val

    def peek(self) -> Any:
        """Returns the front element without removing it."""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        assert self._head is not None
        return self._head.value

    def is_empty(self) -> bool:
        return self._head is None

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return not self.is_empty()


# ==============================================================================
# 3. CORE DATA STRUCTURE: MAX-HEAP (PRIORITY QUEUE)
# ==============================================================================

class HeapItem:
    """Container for heap items with score-based comparison."""
    def __init__(self, priority: float, data: Any):
        self.priority = priority
        self.data = data

    def __lt__(self, other: HeapItem) -> bool:
        return self.priority < other.priority

    def __gt__(self, other: HeapItem) -> bool:
        return self.priority > other.priority

    def __repr__(self) -> str:
        return f"HeapItem(score={self.priority:.4f}, data={self.data})"


class MaxHeap:
    """
    Array-backed binary Max-Heap built from scratch.
    Used to retrieve the Top-K highest-ranked search results.
    Time Complexity:
      - Insert:      O(log N)
      - Extract Max: O(log N)
      - Peek Max:    O(1)
    """
    def __init__(self) -> None:
        self._heap: List[HeapItem] = []

    def push(self, priority: float, data: Any) -> None:
        """Inserts an element with a given priority into the max-heap."""
        item = HeapItem(priority, data)
        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> HeapItem:
        """Extracts and returns the maximum priority item."""
        if self.is_empty():
            raise IndexError("pop from empty heap")
        max_item = self._heap[0]
        last_item = self._heap.pop()
        if self._heap:
            self._heap[0] = last_item
            self._sift_down(0)
        return max_item

    def peek(self) -> HeapItem:
        """Returns the maximum priority item without removing it."""
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def _sift_up(self, idx: int) -> None:
        parent = (idx - 1) // 2
        while idx > 0 and self._heap[idx] > self._heap[parent]:
            self._heap[idx], self._heap[parent] = self._heap[parent], self._heap[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx: int) -> None:
        size = len(self._heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx

            if left < size and self._heap[left] > self._heap[largest]:
                largest = left
            if right < size and self._heap[right] > self._heap[largest]:
                largest = right

            if largest != idx:
                self._heap[idx], self._heap[largest] = self._heap[largest], self._heap[idx]
                idx = largest
            else:
                break


# ==============================================================================
# 4. CORE DATA STRUCTURE: BINARY SEARCH TREE (BST)
# ==============================================================================

class BSTNode:
    """Node for Document Catalog Binary Search Tree."""
    def __init__(self, doc_id: int, title: str, content: str):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.word_count = len(content.split())
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None


class DocumentBST:
    """
    Binary Search Tree to store and maintain document catalog sorted by Doc ID.
    Supports O(log N) lookup (average) and in-order traversal.
    """
    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
        self._count: int = 0

    def insert(self, doc_id: int, title: str, content: str) -> None:
        """Inserts a document into the BST."""
        new_node = BSTNode(doc_id, title, content)
        if self.root is None:
            self.root = new_node
            self._count += 1
            return

        curr = self.root
        while True:
            if doc_id < curr.doc_id:
                if curr.left is None:
                    curr.left = new_node
                    self._count += 1
                    break
                curr = curr.left
            elif doc_id > curr.doc_id:
                if curr.right is None:
                    curr.right = new_node
                    self._count += 1
                    break
                curr = curr.right
            else:
                # Update existing document content
                curr.title = title
                curr.content = content
                curr.word_count = len(content.split())
                break

    def search(self, doc_id: int) -> Optional[BSTNode]:
        """Finds and returns a document node by doc_id."""
        curr = self.root
        while curr:
            if doc_id == curr.doc_id:
                return curr
            elif doc_id < curr.doc_id:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def in_order(self) -> Generator[BSTNode, None, None]:
        """Generator yielding documents in ascending order of doc_id."""
        # Implemented using our custom Stack for iterative traversal
        stack = Stack()
        curr = self.root
        while curr is not None or not stack.is_empty():
            while curr is not None:
                stack.push(curr)
                curr = curr.left
            curr = stack.pop()
            yield curr
            curr = curr.right

    def __len__(self) -> int:
        return self._count


# ==============================================================================
# 5. CORE DATA STRUCTURE: LOW-LEVEL TRIE WITH AUTOCOMPLETE
# ==============================================================================

class TrieNode:
    """
    Low-level node for prefix tree (Trie).
    Maintains character mappings, terminal flag, global corpus frequency,
    and references to matching document IDs.
    """
    def __init__(self, char: str = "") -> None:
        self.char: str = char
        # Dictionary mapping char -> TrieNode
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.frequency: int = 0               # Term frequency in corpus
        self.doc_ids: Set[int] = set()        # Set of documents containing this word


class Trie:
    """
    Low-level Trie implementation for prefix search and auto-complete.
    Time Complexity:
      - Insert:       O(L) where L is length of word
      - Search:       O(L)
      - Starts With:  O(L)
      - Autocomplete: O(L + K) where K is number of matching descendants
    """
    def __init__(self) -> None:
        self.root = TrieNode()
        self.word_count: int = 0

    def insert(self, word: str, doc_id: Optional[int] = None) -> None:
        """
        Inserts a normalized word into the trie.
        Associates doc_id and updates total frequency.
        """
        if not word:
            return

        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode(ch)
            curr = curr.children[ch]

        if not curr.is_end_of_word:
            curr.is_end_of_word = True
            self.word_count += 1

        curr.frequency += 1
        if doc_id is not None:
            curr.doc_ids.add(doc_id)

    def search(self, word: str) -> bool:
        """Returns True if the exact word exists in the Trie."""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Returns True if any word in the Trie begins with prefix."""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Navigates to and returns the TrieNode matching prefix, if present."""
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return None
            curr = curr.children[ch]
        return curr

    def autocomplete(self, prefix: str, max_results: int = 5) -> List[Tuple[str, int, int]]:
        """
        Low-level prefix auto-complete traversal.
        Uses our custom Stack for iterative Depth-First Search (DFS)
        from the prefix node down to all leaf words.

        Returns:
            List of tuples: (suggested_word, corpus_frequency, document_count)
            Sorted by frequency descending, then alphabetically.
        """
        prefix = prefix.strip().lower()
        if not prefix:
            return []

        start_node = self._find_node(prefix)
        if start_node is None:
            return []

        candidates: List[Tuple[str, int, int]] = []

        # DFS using our custom Stack: holds (TrieNode, accumulated_prefix_string)
        traversal_stack = Stack()
        traversal_stack.push((start_node, prefix))

        while not traversal_stack.is_empty():
            node, current_word = traversal_stack.pop()

            if node.is_end_of_word:
                candidates.append((current_word, node.frequency, len(node.doc_ids)))

            # Push children to stack (reverse sorted for predictable traversal)
            for ch in sorted(node.children.keys(), reverse=True):
                traversal_stack.push((node.children[ch], current_word + ch))

        # Rank candidates: higher frequency first, then shorter word, then alpha
        candidates.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        return candidates[:max_results]

    def get_doc_ids(self, word: str) -> Set[int]:
        """Returns set of document IDs containing the exact word."""
        node = self._find_node(word.strip().lower())
        if node and node.is_end_of_word:
            return set(node.doc_ids)
        return set()

    def print_ascii_tree(self, prefix: str = "", max_depth: int = 3) -> None:
        """Renders an ASCII visualization of the Trie starting at prefix."""
        start_node = self._find_node(prefix) if prefix else self.root
        if start_node is None:
            print(f"Prefix '{prefix}' not found in Trie.")
            return

        print(f"\n[Trie Subtree Visualization for prefix: '{prefix}']")
        self._print_tree_recursive(start_node, "", 0, max_depth)
        print()

    def _print_tree_recursive(self, node: TrieNode, indent: str, depth: int, max_depth: int) -> None:
        if depth > max_depth:
            print(f"{indent}... [depth limit reached]")
            return

        for ch, child in sorted(node.children.items()):
            marker = f" (freq={child.frequency}, docs={list(child.doc_ids)})" if child.is_end_of_word else ""
            print(f"{indent}+-- [{ch}]{marker}")
            self._print_tree_recursive(child, indent + "|   ", depth + 1, max_depth)


# ==============================================================================
# 6. BOOLEAN SEARCH ENGINE USING STACK (SHUNting-YARD & RPN)
# ==============================================================================

class BooleanQueryParser:
    """
    Parses and evaluates Boolean Search queries such as:
      '(python AND search) OR (tree AND NOT graph)'

    Employs:
      - Custom Stack for the Shunting-Yard algorithm (Infix -> Reverse Polish Notation).
      - Custom Stack for Postfix expression evaluation over Posting Lists (Sets of Doc IDs).
    """

    OPERATORS = {
        'NOT': {'precedence': 3, 'unary': True},
        'AND': {'precedence': 2, 'unary': False},
        'OR':  {'precedence': 1, 'unary': False},
    }

    @staticmethod
    def tokenize(query_str: str) -> List[str]:
        """Tokenizes boolean query into parentheses, operators, and terms."""
        # Insert spaces around parentheses
        formatted = query_str.replace('(', ' ( ').replace(')', ' ) ')
        raw_tokens = formatted.split()
        tokens: List[str] = []
        for t in raw_tokens:
            upper_t = t.upper()
            if upper_t in ('AND', 'OR', 'NOT', '(', ')'):
                tokens.append(upper_t)
            else:
                clean_term = re.sub(r'[^a-zA-Z0-9_-]', '', t).lower()
                if clean_term:
                    tokens.append(clean_term)
        return tokens

    @classmethod
    def infix_to_postfix(cls, tokens: List[str]) -> List[str]:
        """
        Shunting-Yard algorithm using our custom Stack.
        Converts infix expression to Postfix / Reverse Polish Notation (RPN).
        """
        output: List[str] = []
        op_stack = Stack()

        for token in tokens:
            if token in cls.OPERATORS:
                token_prec = cls.OPERATORS[token]['precedence']
                while not op_stack.is_empty() and op_stack.peek() != '(':
                    top_op = op_stack.peek()
                    if top_op in cls.OPERATORS and cls.OPERATORS[top_op]['precedence'] >= token_prec:
                        output.append(op_stack.pop())
                    else:
                        break
                op_stack.push(token)

            elif token == '(':
                op_stack.push(token)

            elif token == ')':
                found_left_paren = False
                while not op_stack.is_empty():
                    top = op_stack.pop()
                    if top == '(':
                        found_left_paren = True
                        break
                    output.append(top)
                if not found_left_paren:
                    raise ValueError("Mismatched parentheses in search query.")

            else:
                # Operand (search term)
                output.append(token)

        while not op_stack.is_empty():
            top = op_stack.pop()
            if top in ('(', ')'):
                raise ValueError("Mismatched parentheses in search query.")
            output.append(top)

        return output

    @classmethod
    def evaluate_rpn(cls, postfix_tokens: List[str], trie: Trie, all_doc_ids: Set[int]) -> Set[int]:
        """
        Evaluates Postfix tokens using our custom Stack.
        Operands resolve to sets of matching doc_ids.
        Operators execute set operations:
          - AND : Intersection
          - OR  : Union
          - NOT : Difference from universe (all_doc_ids)
        """
        eval_stack = Stack()

        for token in postfix_tokens:
            if token == 'NOT':
                if eval_stack.is_empty():
                    raise ValueError("Syntax error: NOT operator missing operand.")
                operand: Set[int] = eval_stack.pop()
                # Difference: universe - operand
                eval_stack.push(all_doc_ids - operand)

            elif token in ('AND', 'OR'):
                if len(eval_stack) < 2:
                    raise ValueError(f"Syntax error: {token} operator requires two operands.")
                right_set: Set[int] = eval_stack.pop()
                left_set: Set[int] = eval_stack.pop()

                if token == 'AND':
                    eval_stack.push(left_set & right_set)
                elif token == 'OR':
                    eval_stack.push(left_set | right_set)

            else:
                # Operand term: fetch postings from Trie
                term_docs = trie.get_doc_ids(token)
                eval_stack.push(term_docs)

        if len(eval_stack) != 1:
            raise ValueError("Malformed boolean expression.")

        return eval_stack.pop()


# ==============================================================================
# 7. DOCUMENT SEARCH ENGINE SYSTEM
# ==============================================================================

class DocumentSearchEngine:
    """
    Integrated Document Search Engine coordinating all data structures:
      - Queue: document ingestion pipeline
      - BST: document repository
      - Trie: low-level autocomplete & prefix index
      - Inverted Index: term frequency mapping for TF-IDF
      - Max-Heap: top-k ranked retrieval
      - Stack: boolean query evaluation
    """

    STOP_WORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
        'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
        'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into',
        'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my',
        'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
        'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
        'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the',
        'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
        'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
        'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
        'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself'
    }

    def __init__(self) -> None:
        self.doc_queue = Queue()            # Ingestion Queue
        self.doc_bst = DocumentBST()        # Document Catalog BST
        self.trie = Trie()                  # Autocomplete Trie
        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.all_doc_ids: Set[int] = set()
        self._next_id: int = 1

    def queue_document(self, title: str, content: str) -> int:
        """Enqueues a new document into the processing pipeline."""
        assigned_id = self._next_id
        self._next_id += 1
        self.doc_queue.enqueue((assigned_id, title, content))
        return assigned_id

    def process_ingestion_queue(self) -> int:
        """
        Dequeues documents from the FIFO queue and indexes them:
          1. Inserts into DocumentBST
          2. Extracts words and inserts into Trie
          3. Updates inverted index postings
        """
        processed_count = 0
        while not self.doc_queue.is_empty():
            doc_id, title, content = self.doc_queue.dequeue()

            # 1. Insert into Document BST
            self.doc_bst.insert(doc_id, title, content)
            self.all_doc_ids.add(doc_id)

            # 2. Tokenize text
            tokens = self._tokenize(title + " " + content)
            term_freqs: Dict[str, int] = {}

            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
                # Insert into low-level Trie
                self.trie.insert(token, doc_id)

            # 3. Update Inverted Index postings
            for term, count in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count

            processed_count += 1

        return processed_count

    def _tokenize(self, text: str) -> List[str]:
        """Low-level tokenization: lowercase, regex clean, filter empty."""
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [w for w in words if len(w) > 1]

    def autocomplete(self, prefix: str, max_results: int = 5) -> List[Tuple[str, int, int]]:
        """Delegates auto-complete to low-level Trie."""
        return self.trie.autocomplete(prefix, max_results=max_results)

    def search_boolean(self, query: str) -> List[BSTNode]:
        """
        Executes a Boolean Search query using Stacks (Shunting-Yard + Postfix evaluation).
        Supports: AND, OR, NOT, ( )
        """
        tokens = BooleanQueryParser.tokenize(query)
        if not tokens:
            return []

        # Convert to RPN postfix using custom Stack
        postfix = BooleanQueryParser.infix_to_postfix(tokens)

        # Evaluate RPN using custom Stack
        matching_ids = BooleanQueryParser.evaluate_rpn(postfix, self.trie, self.all_doc_ids)

        # Fetch document nodes from BST
        results: List[BSTNode] = []
        for doc_id in sorted(matching_ids):
            node = self.doc_bst.search(doc_id)
            if node:
                results.append(node)
        return results

    def search_ranked(self, query: str, top_k: int = 5) -> List[Tuple[BSTNode, float, List[str]]]:
        """
        Ranked retrieval using TF-IDF and a custom Max-Heap.
        Formula:
          TF(t, d) = count(t, d) / total_words(d)
          IDF(t)   = log(1 + (Total_Docs / (1 + Docs_With_t)))
          Score(d) = sum(TF * IDF for each query term t)
        """
        query_terms = [t for t in self._tokenize(query) if t not in self.STOP_WORDS]
        if not query_terms:
            # Fallback to all terms if all were stop words
            query_terms = self._tokenize(query)
        if not query_terms:
            return []

        total_docs = len(self.all_doc_ids)
        if total_docs == 0:
            return []

        doc_scores: Dict[int, float] = {}
        matched_terms_per_doc: Dict[int, Set[str]] = {}

        for term in query_terms:
            postings = self.inverted_index.get(term, {})
            df = len(postings)
            idf = math.log(1.0 + (total_docs / (1.0 + df)))

            for doc_id, count in postings.items():
                doc_node = self.doc_bst.search(doc_id)
                doc_len = doc_node.word_count if doc_node else 100
                tf = count / max(1, doc_len)
                score = tf * idf

                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                if doc_id not in matched_terms_per_doc:
                    matched_terms_per_doc[doc_id] = set()
                matched_terms_per_doc[doc_id].add(term)

        # Use our custom Max-Heap to extract top-k scored documents
        heap = MaxHeap()
        for doc_id, score in doc_scores.items():
            heap.push(score, doc_id)

        ranked_results: List[Tuple[BSTNode, float, List[str]]] = []
        while not heap.is_empty() and len(ranked_results) < top_k:
            item = heap.pop()
            doc_id = item.data
            doc_node = self.doc_bst.search(doc_id)
            if doc_node:
                matched = sorted(list(matched_terms_per_doc.get(doc_id, set())))
                ranked_results.append((doc_node, item.priority, matched))

        return ranked_results

    def generate_snippet(self, content: str, highlight_terms: List[str], max_len: int = 160) -> str:
        """Generates a snippet highlighting occurrences of matching terms."""
        clean_terms = {t.lower() for t in highlight_terms if t}
        words = content.split()
        if not words:
            return ""

        # Find first matching word index
        best_idx = 0
        for i, w in enumerate(words):
            stripped = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if stripped in clean_terms:
                best_idx = max(0, i - 5)
                break

        selected_words = words[best_idx:best_idx + 25]
        snippet_words = []
        for w in selected_words:
            stripped = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if stripped in clean_terms:
                snippet_words.append(f"**{w}**")
            else:
                snippet_words.append(w)

        snippet = " ".join(snippet_words)
        if best_idx > 0:
            snippet = "... " + snippet
        if best_idx + 25 < len(words):
            snippet = snippet + " ..."
        return snippet


# ==============================================================================
# 8. PRE-LOADED SAMPLE CORPUS
# ==============================================================================

SAMPLE_DOCUMENTS = [
    (
        "Trie Data Structure & Prefix Trees",
        "A Trie, also called a prefix tree, is an ordered tree data structure used to store an "
        "associative array where the keys are usually strings. Unlike a binary search tree, no node "
        "in the tree stores the key associated with that node; instead, its position in the tree "
        "defines the key with which it is associated. Tries are remarkably fast for prefix auto-complete, "
        "spell checking, and IP routing lookup tables."
    ),
    (
        "Stack Data Structure and Applications",
        "A stack is a linear data structure that follows the Last-In First-Out (LIFO) principle. "
        "Common operations are push, pop, and peek. Stacks are fundamental in computer science, "
        "powering expression evaluation such as Shunting-Yard algorithm for infix to postfix conversion, "
        "syntax parsing, recursion call stacks, and backtracking algorithms."
    ),
    (
        "Binary Search Trees and Balanced Trees",
        "A Binary Search Tree (BST) is a node-based binary tree data structure where each node has at "
        "most two children. The left subtree contains keys less than the node, and the right subtree "
        "contains keys greater than the node. Balanced variants like AVL trees and Red-Black trees "
        "guarantee logarithmic time complexity O(log N) for search, insertion, and deletion."
    ),
    (
        "Queue Data Structure & FIFO Processing",
        "A queue is a First-In First-Out (FIFO) data structure where elements are inserted at the rear "
        "and removed from the front. Queues are essential for scheduling algorithms, breadth-first search "
        "(BFS) graph traversal, asynchronous task pipelines, and document ingestion streaming."
    ),
    (
        "Priority Queues and Binary Heaps",
        "A priority queue is an abstract data-type where each element has an associated priority. "
        "Binary heaps, including Min-Heap and Max-Heap, provide efficient O(log N) insertion and "
        "O(1) peek for the extreme element. Heaps are widely utilized in Dijkstra's shortest path, "
        "Huffman coding, and search engine Top-K relevance ranking algorithms."
    ),
    (
        "Modern Search Engines and Inverted Indexes",
        "Search engines index billions of web documents using inverted index structures. "
        "An inverted index maps every distinct vocabulary term to a posting list containing document IDs "
        "and term frequency positions. Scoring algorithms like TF-IDF and BM25 compute relevance scores "
        "between user search queries and indexed documents."
    ),
    (
        "Graph Traversal: BFS vs DFS",
        "Graphs consist of vertices connected by edges. Breadth-First Search (BFS) explores neighbor nodes "
        "layer by layer using a FIFO queue, finding shortest paths in unweighted graphs. "
        "Depth-First Search (DFS) explores as deep as possible along each branch using recursion or a LIFO stack."
    ),
    (
        "Autocomplete Systems in Web Search",
        "Autocomplete prediction assists users by suggesting query completions in real-time as they type. "
        "Trie data structures enhanced with frequency heaps enable lightning-fast prefix lookups under "
        "strict latency budgets, sorting popular queries ahead of obscure terms."
    ),
]


def load_sample_corpus(engine: DocumentSearchEngine) -> None:
    """Populates the search engine with sample computer science documents."""
    for title, content in SAMPLE_DOCUMENTS:
        engine.queue_document(title, content)
    processed = engine.process_ingestion_queue()
    print(f"[*] Successfully queued and indexed {processed} sample documents.")


def load_documents_from_folder(engine: DocumentSearchEngine, folder_path: str = "documents") -> int:
    """
    Crawls a local folder, queues all files into the FIFO ingestion Queue,
    and processes the indexing pipeline (BST, Trie, and Inverted Index).
    Supports .txt, .md, .log, and extensionless text files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # Gather readable files (ignore hidden files starting with '.')
    files = [
        f for f in folder.iterdir()
        if f.is_file() and not f.name.startswith('.')
        and (not f.suffix or f.suffix.lower() in ('.txt', '.md', '.log', '.csv', '.text', '.py'))
    ]

    # If folder is empty, seed with starter documents and test files
    if not files:
        print(f"[*] Folder '{folder_path}' is empty. Seeding starter text documents...")
        starter_files = [
            ("cat.txt", "A domestic cat is a small carnivorous mammal kept as a beloved pet."),
            ("dog.txt", "A domestic dog is a loyal companion animal and household pet."),
            ("horse.txt", "A horse is a large domesticated four-legged mammal adapted to riding."),
            ("hotdog.txt", "A hotdog is a food consisting of a grilled sausage in a sliced bun."),
            ("cation.txt", "A cation is a positively charged ion with fewer electrons than protons."),
            ("scatman.txt", "Scatman John was an iconic musician famous for scat singing style."),
        ]
        # Include sample CS documents as well
        for title, content in SAMPLE_DOCUMENTS:
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', title).strip('_').lower() + ".txt"
            starter_files.append((safe_name, content))

        for fname, fcontent in starter_files:
            (folder / fname).write_text(fcontent, encoding='utf-8')

        files = [
            f for f in folder.iterdir()
            if f.is_file() and not f.name.startswith('.')
            and (not f.suffix or f.suffix.lower() in ('.txt', '.md', '.log', '.csv', '.text', '.py'))
        ]

    queued_count = 0
    for file_path in sorted(files, key=lambda p: p.name.lower()):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore').strip()
            # If the file has content, use it; otherwise, use filename as content
            doc_content = content if content else file_path.stem
            engine.queue_document(file_path.name, doc_content)
            queued_count += 1
        except Exception as e:
            print(f"  [!] Could not read '{file_path.name}': {e}")

    processed = engine.process_ingestion_queue()
    print(f"[*] Successfully crawled and indexed {processed} document(s) from '{folder_path}/'.")
    return processed


# ==============================================================================
# 9. SELF-TESTING & VALIDATION SUITE
# ==============================================================================

def run_automated_tests() -> None:
    """Executes verification tests for all custom data structures."""
    print("==================================================================")
    print(" RUNNING COMPREHENSIVE DATA STRUCTURE SELF-TESTS")
    print("==================================================================")

    # 1. Test Stack
    print("[1/6] Testing Custom Stack (LIFO)...", end=" ")
    s = Stack()
    assert s.is_empty(), "Stack should initially be empty"
    s.push(10)
    s.push(20)
    s.push(30)
    assert len(s) == 3
    assert s.peek() == 30
    assert s.pop() == 30
    assert s.pop() == 20
    assert s.pop() == 10
    assert s.is_empty()
    print("PASSED")

    # 2. Test Queue
    print("[2/6] Testing Custom Queue (FIFO)...", end=" ")
    q = Queue()
    assert q.is_empty()
    q.enqueue("apple")
    q.enqueue("banana")
    q.enqueue("cherry")
    assert len(q) == 3
    assert q.peek() == "apple"
    assert q.dequeue() == "apple"
    assert q.dequeue() == "banana"
    assert q.dequeue() == "cherry"
    assert q.is_empty()
    print("PASSED")

    # 3. Test MaxHeap
    print("[3/6] Testing Custom Max-Heap (Priority Queue)...", end=" ")
    heap = MaxHeap()
    heap.push(1.5, "doc1")
    heap.push(5.2, "doc2")
    heap.push(3.1, "doc3")
    heap.push(0.8, "doc4")
    assert heap.pop().data == "doc2"  # score 5.2
    assert heap.pop().data == "doc3"  # score 3.1
    assert heap.pop().data == "doc1"  # score 1.5
    assert heap.pop().data == "doc4"  # score 0.8
    assert heap.is_empty()
    print("PASSED")

    # 4. Test Trie & Autocomplete
    print("[4/6] Testing Low-Level Trie & Autocomplete...", end=" ")
    trie = Trie()
    trie.insert("search", doc_id=1)
    trie.insert("search", doc_id=2)  # repeated word increases frequency
    trie.insert("searching", doc_id=1)
    trie.insert("sea", doc_id=3)
    trie.insert("season", doc_id=4)
    trie.insert("stack", doc_id=5)

    assert trie.search("search") is True
    assert trie.search("sear") is False
    assert trie.starts_with("sea") is True

    # Check autocomplete suggestions for prefix 'sea'
    suggestions = trie.autocomplete("sea", max_results=5)
    # Expected words: 'search' (freq=2), 'sea', 'searching', 'season'
    suggested_words = [item[0] for item in suggestions]
    assert "search" in suggested_words
    assert suggested_words[0] == "search", "Highest frequency term should rank first"
    print("PASSED")

    # 5. Test Document BST
    print("[5/6] Testing Document Binary Search Tree...", end=" ")
    bst = DocumentBST()
    bst.insert(5, "Doc 5", "Five")
    bst.insert(2, "Doc 2", "Two")
    bst.insert(8, "Doc 8", "Eight")
    bst.insert(1, "Doc 1", "One")

    in_order_ids = [node.doc_id for node in bst.in_order()]
    assert in_order_ids == [1, 2, 5, 8], f"Expected [1, 2, 5, 8], got {in_order_ids}"
    assert bst.search(8) is not None
    assert bst.search(99) is None
    print("PASSED")

    # 6. Test Shunting-Yard Boolean Query Parser
    print("[6/6] Testing Stack-based Shunting-Yard & Boolean Search...", end=" ")
    engine = DocumentSearchEngine()
    load_sample_corpus(engine)

    # Boolean query: '(stack AND lifo) OR trie'
    results = engine.search_boolean("(stack AND lifo) OR trie")
    doc_titles = [r.title for r in results]
    assert any("Stack" in t for t in doc_titles)
    assert any("Trie" in t for t in doc_titles)

    # Test NOT operator: 'tree AND NOT binary'
    not_results = engine.search_boolean("tree AND NOT binary")
    for r in not_results:
        assert "Binary Search Trees" not in r.title, "Should exclude Binary Search Trees"

    print("PASSED")
    print("==================================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! Data structures functioning properly.")
    print("==================================================================\n")


# ==============================================================================
# 10. INTERACTIVE USER INTERFACE (CLI)
# ==============================================================================

def print_banner() -> None:
    banner = r"""
+============================================================================+
|                   DOCUMENT SEARCH ENGINE & TRIE AUTOCOMPLETE               |
|  Data Structures: Trie (Prefixes) | Stack (Boolean) | Queue (Ingestion)    |
|                   MaxHeap (Ranked) | BST (Catalog)   | Inverted Index       |
+============================================================================+
"""
    print(banner)


def display_menu() -> None:
    print("\n------------------------------ MAIN MENU ------------------------------")
    print("  1. Low-Level Auto-Complete (Trie Prefix Search & Suggestions)")
    print("  2. Boolean Document Search (Stack-based: AND, OR, NOT, Parentheses)")
    print("  3. Ranked Keyword Search   (TF-IDF with Max-Heap Top-K Retrieval)")
    print("  4. Ingest Documents        (Scan 'documents/' folder or enter manually)")
    print("  5. View All Documents      (BST In-Order Traversal)")
    print("  6. Inspect Data Structures (Trie Visualizer & System Statistics)")
    print("  7. Run Test Suite          (Automated Verification of All DS)")
    print("  8. Exit")
    print("------------------------------------------------------------------------")


def handle_autocomplete(engine: DocumentSearchEngine) -> None:
    print("\n[LOW-LEVEL TRIE AUTO-COMPLETE]")
    print("Type any prefix (e.g. 'tr', 'sta', 'sear', 'alg', 'qu') to see suggestions.")
    print("Type ':q' to return to menu.\n")

    while True:
        prefix = input("Enter prefix: ").strip()
        if prefix.lower() == ':q':
            break
        if not prefix:
            continue

        results = engine.autocomplete(prefix, max_results=8)
        if not results:
            print(f"  [x] No completions found for prefix: '{prefix}'")
        else:
            print(f"\n  Found {len(results)} suggestions for '{prefix}':")
            print("  +------+------------------------+-------------+--------------+")
            print("  | Rank | Suggested Term         | Corpus Freq | Document Qty |")
            print("  +------+------------------------+-------------+--------------+")
            for rank, (word, freq, doc_count) in enumerate(results, start=1):
                print(f"  | {rank:<4} | {word:<22} | {freq:<11} | {doc_count:<12} |")
            print("  +------+------------------------+-------------+--------------+")
            matching_docs = engine.trie.get_doc_ids(results[0][0])
            print(f"  Top pick '{results[0][0]}' is present in Doc IDs: {sorted(list(matching_docs))}\n")


def handle_boolean_search(engine: DocumentSearchEngine) -> None:
    print("\n[BOOLEAN SEARCH USING STACKS]")
    print("Supported syntax:")
    print("  - Operators  : AND, OR, NOT")
    print("  - Grouping   : Parentheses ( ... )")
    print("  - Examples   :")
    print("      trie AND prefix")
    print("      (stack OR queue) AND traversal")
    print("      search AND NOT web")
    print("Type ':q' to return to menu.\n")

    while True:
        query = input("Boolean Query: ").strip()
        if query.lower() == ':q':
            break
        if not query:
            continue

        try:
            tokens = BooleanQueryParser.tokenize(query)
            postfix = BooleanQueryParser.infix_to_postfix(tokens)
            print(f"  [Stack Parser] Tokens  : {tokens}")
            print(f"  [Shunting-Yard] Postfix: {postfix}")

            results = engine.search_boolean(query)
            if not results:
                print("  [x] No documents matched the boolean criteria.\n")
            else:
                print(f"\n  [OK] {len(results)} document(s) matched:")
                for node in results:
                    terms_to_highlight = [t for t in tokens if t not in ('AND', 'OR', 'NOT', '(', ')')]
                    snippet = engine.generate_snippet(node.content, terms_to_highlight)
                    print(f"  +-- [Doc #{node.doc_id}] {node.title}")
                    print(f"  |   Words: {node.word_count}")
                    print(f"  |   Snippet: {snippet}")
                    print(f"  +-----------------------------------------------------")
                print()
        except Exception as e:
            print(f"  [!] Query Error: {e}\n")


def handle_ranked_search(engine: DocumentSearchEngine) -> None:
    print("\n[RANKED TF-IDF SEARCH WITH MAX-HEAP]")
    print("Enter search keywords (e.g., 'tree traversal algorithms', 'fast prefix lookup').")
    print("Type ':q' to return to menu.\n")

    while True:
        query = input("Search query: ").strip()
        if query.lower() == ':q':
            break
        if not query:
            continue

        results = engine.search_ranked(query, top_k=5)
        if not results:
            print("  [x] No matching documents found.\n")
        else:
            print(f"\n  Top {len(results)} Results (Ranked by Max-Heap TF-IDF Score):")
            for rank, (node, score, matched) in enumerate(results, start=1):
                snippet = engine.generate_snippet(node.content, matched)
                print(f"  [{rank}] Score: {score:.5f} | Doc #{node.doc_id}: {node.title}")
                print(f"      Matched terms : {', '.join(matched)}")
                print(f"      Snippet       : {snippet}")
            print()


def handle_add_document(engine: DocumentSearchEngine) -> None:
    print("\n[DOCUMENT INGESTION PIPELINE]")
    print("  1. Scan and index files from 'documents/' folder (Default)")
    print("  2. Manually type a new document title and content")
    choice = input("Select option [1/2] (default 1): ").strip()

    if choice == '2':
        title = input("Document Title: ").strip()
        if not title:
            print("Title cannot be empty.")
            return
        print("Enter Document Content (one line or paragraph):")
        content = input("> ").strip()
        if not content:
            print("Content cannot be empty.")
            return

        doc_id = engine.queue_document(title, content)
        print(f"\n  [Queue] Document #{doc_id} enqueued into Ingestion FIFO Queue.")
        processed = engine.process_ingestion_queue()
        print(f"  [Pipeline] Ingestion queue processed {processed} item(s).")
        print(f"  [Index] Document #{doc_id} successfully mapped to BST, Trie, and Inverted Index.\n")
    else:
        print("\n[*] Scanning 'documents/' subfolder...")
        load_documents_from_folder(engine, "documents")


def handle_view_catalog(engine: DocumentSearchEngine) -> None:
    print("\n[DOCUMENT CATALOG (BST In-Order Traversal)]")
    count = 0
    print("=" * 70)
    for node in engine.doc_bst.in_order():
        count += 1
        print(f"Doc ID #{node.doc_id:<3} | Words: {node.word_count:<4} | Title: {node.title}")
    print("=" * 70)
    print(f"Total documents stored in BST: {count}\n")


def handle_inspect_ds(engine: DocumentSearchEngine) -> None:
    print("\n[DATA STRUCTURE INSPECTOR & METRICS]")
    print(f"  • Total Indexed Documents (BST)   : {len(engine.doc_bst)}")
    print(f"  • Distinct Words in Trie          : {engine.trie.word_count}")
    print(f"  • Inverted Index Vocabulary Size  : {len(engine.inverted_index)}")
    print(f"  • Current Ingestion Queue Size    : {len(engine.doc_queue)}")

    choice = input("\nWould you like to print an ASCII subtree of the Trie? (y/n): ").strip().lower()
    if choice == 'y':
        prefix = input("Enter prefix to visualize (or press Enter for root): ").strip()
        engine.trie.print_ascii_tree(prefix=prefix, max_depth=2)


def main() -> None:
    """Main program entry point."""
    print_banner()

    engine = DocumentSearchEngine()
    print("[*] Initializing search engine & scanning 'documents/' folder...")
    load_documents_from_folder(engine, "documents")

    # Allow CLI flag --test for quick non-interactive verification
    if "--test" in sys.argv:
        run_automated_tests()
        return

    while True:
        display_menu()
        choice = input("Select an option [1-8]: ").strip()

        if choice == '1':
            handle_autocomplete(engine)
        elif choice == '2':
            handle_boolean_search(engine)
        elif choice == '3':
            handle_ranked_search(engine)
        elif choice == '4':
            handle_add_document(engine)
        elif choice == '5':
            handle_view_catalog(engine)
        elif choice == '6':
            handle_inspect_ds(engine)
        elif choice == '7':
            run_automated_tests()
        elif choice == '8':
            print("\nThank you for using the Document Search Engine. Goodbye!\n")
            break
        else:
            print("[!] Invalid selection. Please enter a number between 1 and 8.")


if __name__ == '__main__':
    main()
