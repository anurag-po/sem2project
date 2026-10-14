"""
gui.py
======
Tkinter GUI for the Document Search Engine.
Run:  python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from document_search_engine import DocumentSearchEngine, load_documents_from_folder


class SearchEngineGUI:
    """Simple Tkinter front-end for the Document Search Engine."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Document Search Engine")
        self.root.geometry("860x620")
        self.root.minsize(700, 500)

        # -- Engine ----------------------------------------------------------
        self.engine = DocumentSearchEngine()
        load_documents_from_folder(self.engine, "documents")

        # -- Notebook (tabs) -------------------------------------------------
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_autocomplete_tab()
        self._build_boolean_tab()
        self._build_ranked_tab()
        self._build_add_doc_tab()
        self._build_catalog_tab()

    # ================================================================
    # Tab 1 : Auto-Complete
    # ================================================================
    def _build_autocomplete_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Auto-Complete  ")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=(10, 4))

        ttk.Label(top, text="Prefix:").pack(side=tk.LEFT)
        self.ac_entry = ttk.Entry(top, width=30)
        self.ac_entry.pack(side=tk.LEFT, padx=6)
        self.ac_entry.bind("<KeyRelease>", self._on_prefix_key)

        ttk.Button(top, text="Suggest", command=self._run_autocomplete).pack(side=tk.LEFT)

        self.ac_output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED)
        self.ac_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

    def _on_prefix_key(self, _event: tk.Event) -> None:
        self._run_autocomplete()

    def _run_autocomplete(self) -> None:
        prefix = self.ac_entry.get().strip()
        self._clear(self.ac_output)
        if not prefix:
            self._append(self.ac_output, "Type a prefix to see suggestions...")
            return

        results = self.engine.autocomplete(prefix, max_results=10)
        if not results:
            self._append(self.ac_output, f"No completions found for '{prefix}'.")
            return

        header = f"{'Rank':<6}{'Suggested Term':<24}{'Frequency':<12}{'Documents'}\n"
        header += "-" * 56 + "\n"
        self._append(self.ac_output, header)

        for rank, (word, freq, doc_count) in enumerate(results, 1):
            line = f"{rank:<6}{word:<24}{freq:<12}{doc_count}\n"
            self._append(self.ac_output, line)

        doc_ids = self.engine.trie.get_doc_ids(results[0][0])
        self._append(self.ac_output, f"\nTop suggestion '{results[0][0]}' appears in doc IDs: {sorted(doc_ids)}")

    # ================================================================
    # Tab 2 : Boolean Search
    # ================================================================
    def _build_boolean_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Boolean Search  ")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=(10, 4))

        ttk.Label(top, text="Query:").pack(side=tk.LEFT)
        self.bool_entry = ttk.Entry(top, width=40)
        self.bool_entry.pack(side=tk.LEFT, padx=6)
        self.bool_entry.bind("<Return>", lambda _e: self._run_boolean())

        ttk.Button(top, text="Search", command=self._run_boolean).pack(side=tk.LEFT)

        hint = ttk.Label(tab, text="Operators: AND  OR  NOT    Grouping: ( )    "
                         "Example: (trie OR stack) AND traversal",
                         foreground="gray")
        hint.pack(anchor=tk.W, padx=8)

        self.bool_output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED)
        self.bool_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

    def _run_boolean(self) -> None:
        query = self.bool_entry.get().strip()
        self._clear(self.bool_output)
        if not query:
            return

        try:
            results = self.engine.search_boolean(query)
            if not results:
                self._append(self.bool_output, "No documents matched.")
                return

            self._append(self.bool_output, f"{len(results)} document(s) matched:\n\n")
            for node in results:
                snippet = node.content[:200] + ("..." if len(node.content) > 200 else "")
                block = (f"[Doc #{node.doc_id}]  {node.title}\n"
                         f"  Words: {node.word_count}\n"
                         f"  {snippet}\n\n")
                self._append(self.bool_output, block)
        except Exception as exc:
            self._append(self.bool_output, f"Error: {exc}")

    # ================================================================
    # Tab 3 : Ranked Search
    # ================================================================
    def _build_ranked_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Ranked Search  ")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=(10, 4))

        ttk.Label(top, text="Keywords:").pack(side=tk.LEFT)
        self.rank_entry = ttk.Entry(top, width=40)
        self.rank_entry.pack(side=tk.LEFT, padx=6)
        self.rank_entry.bind("<Return>", lambda _e: self._run_ranked())

        ttk.Button(top, text="Search", command=self._run_ranked).pack(side=tk.LEFT)

        self.rank_output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED)
        self.rank_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

    def _run_ranked(self) -> None:
        query = self.rank_entry.get().strip()
        self._clear(self.rank_output)
        if not query:
            return

        results = self.engine.search_ranked(query, top_k=5)
        if not results:
            self._append(self.rank_output, "No matching documents found.")
            return

        self._append(self.rank_output, f"Top {len(results)} results (TF-IDF via Max-Heap):\n\n")
        for rank, (node, score, matched) in enumerate(results, 1):
            snippet = self.engine.generate_snippet(node.content, matched)
            block = (f"#{rank}  Score: {score:.5f}  |  Doc #{node.doc_id}: {node.title}\n"
                     f"    Matched: {', '.join(matched)}\n"
                     f"    {snippet}\n\n")
            self._append(self.rank_output, block)

    # ================================================================
    # Tab 4 : Add Document
    # ================================================================
    def _build_add_doc_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Add Document  ")

        form = ttk.Frame(tab)
        form.pack(fill=tk.X, padx=8, pady=(10, 4))

        ttk.Label(form, text="Title:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.add_title = ttk.Entry(form, width=60)
        self.add_title.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(form, text="Content:").grid(row=1, column=0, sticky=tk.NW, pady=4)
        self.add_content = tk.Text(form, height=8, width=60, wrap=tk.WORD)
        self.add_content.grid(row=1, column=1, padx=6, pady=4)

        btn_frame = ttk.Frame(form)
        btn_frame.grid(row=2, column=1, sticky=tk.W, padx=6, pady=6)
        ttk.Button(btn_frame, text="Add & Index", command=self._add_document).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Scan & Ingest 'documents/' Folder", command=self._scan_folder).pack(side=tk.LEFT)

        self.add_status = ttk.Label(tab, text="", foreground="green")
        self.add_status.pack(anchor=tk.W, padx=14)

    def _add_document(self) -> None:
        title = self.add_title.get().strip()
        content = self.add_content.get("1.0", tk.END).strip()
        if not title or not content:
            messagebox.showwarning("Missing Fields", "Please enter both a title and content.")
            return

        doc_id = self.engine.queue_document(title, content)
        self.engine.process_ingestion_queue()
        self.add_title.delete(0, tk.END)
        self.add_content.delete("1.0", tk.END)
        self.add_status.config(
            text=f"Document #{doc_id} '{title}' added and indexed successfully.")

    def _scan_folder(self) -> None:
        count = load_documents_from_folder(self.engine, "documents")
        self.add_status.config(
            text=f"Successfully indexed {count} document(s) from 'documents/' folder.")

    # ================================================================
    # Tab 5 : Catalog
    # ================================================================
    def _build_catalog_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Document Catalog  ")

        ttk.Button(tab, text="Refresh Catalog (BST In-Order)",
                   command=self._refresh_catalog).pack(anchor=tk.W, padx=8, pady=(10, 4))

        self.cat_output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED)
        self.cat_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._refresh_catalog()

    def _refresh_catalog(self) -> None:
        self._clear(self.cat_output)
        header = f"{'ID':<6}{'Words':<8}{'Title'}\n" + "-" * 60 + "\n"
        self._append(self.cat_output, header)

        count = 0
        for node in self.engine.doc_bst.in_order():
            self._append(self.cat_output, f"{node.doc_id:<6}{node.word_count:<8}{node.title}\n")
            count += 1

        self._append(self.cat_output, f"\nTotal documents: {count}")
        self._append(self.cat_output, f"\nDistinct words in Trie: {self.engine.trie.word_count}")
        self._append(self.cat_output, f"\nInverted index vocabulary: {len(self.engine.inverted_index)}")

    # ================================================================
    # Helpers
    # ================================================================
    @staticmethod
    def _clear(widget: scrolledtext.ScrolledText) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.config(state=tk.DISABLED)

    @staticmethod
    def _append(widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    SearchEngineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
