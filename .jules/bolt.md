## 2023-10-27 - Vectorized MMR implementation
**Learning:** The MMR (Maximal Marginal Relevance) implementation in `SemanticRouter._apply_mmr` uses a nested loop in pure Python to calculate cosine similarity between the current candidate and all previously selected items (`O(K * |candidates| * |selected|)` dot products).
**Action:** Replace the nested loop with a vectorized approach using numpy. By normalizing the embeddings upfront and calculating similarities via matrix multiplication `np.dot(normed_emb, last_selected_emb)`, we can drastically reduce CPU time. Maintain a running array of `max_sims` to avoid re-evaluating similarities with all previously selected tools at each step.

## 2026-04-16 - Optimize string serialization of embedding vectors
**Learning:** Using Python's built-in `json.dumps()` is significantly faster and safer for serializing large numerical lists (like embeddings) compared to generator expressions with string concatenation (`"[" + ",".join(str(x) for x in embedding) + "]"`), and it is much safer than `str()` because it reliably produces a Postgres array-compatible format (`[1.0, 2.0]`) regardless of whether the input is a list, tuple, or numpy array.
**Action:** Use `json.dumps()` whenever bulk-converting lists to string representations.

## 2023-10-27 - Fast JSON float array serialization
**Learning:** For converting large lists of floats (like vector embeddings) into strings for database queries, `json.dumps(embedding)` is measurably faster (about ~7-8% speedup) and more readable than using a generator expression with string concatenation `("[" + ",".join(str(x) for x in embedding) + "]")`.
**Action:** Use standard `json.dumps` instead of manual string parsing when passing vector data as strings to database drivers like `asyncpg`.

## 2023-10-27 - Fast token matching with substring pre-check
**Learning:** In string parsing functions that use compiled regular expressions (like `_get_token_pattern(token).search(text)`), evaluating the regex engine is expensive. When the token does not exist in the text at all, the regex engine still scans the whole string.
**Action:** Always add a fast substring pre-check (`if token not in text: return False`) before running the regular expression. The native Python `in` operator uses highly optimized C algorithms and provides a massive speedup (>10x in microbenchmarks) for the non-matching case.

## 2024-05-18 - Fast List Filtering with set.issubset
**Learning:** Checking elements inside list comprehensions via `all(...)` or `any(...)` generator expressions creates significant overhead in tight loops (e.g., candidate filtering during semantic routing). The overhead comes from iterating over Python generators instead of native C code.
**Action:** Replace `all(x in y)` and `any(x in y)` with fast native `set` operations like `set.issubset(y)` and `set.isdisjoint(y)`. This yields ~3x speedup. In lists/loops, extract the static side into a `set` *before* the loop.

## 2023-10-27 - Fast regex vs generator expressions for string matching
**Learning:** Using generator expressions like `any(kw in text for kw in keywords)` or `any(ord(c) < 32 for c in text)` is a major performance bottleneck in inner loops due to Python generator overhead.
**Action:** Replace these with pre-compiled regular expressions and `.search()`. This offloads the iteration to optimized C code, yielding >5x speedup for character validation and ~40% speedup for multi-keyword matching. When replacing simple `kw in text` checks, omit word boundaries to preserve the exact substring matching behavior.

## 2026-05-15 - Single-pass maximum tracking in tight loops
**Learning:** In Python, replacing generator expressions combined with dictionary generation and multiple `max()` calls (e.g., `max(scores.values())`) with a single-pass nested loop that tracks the maximum value inline can reduce latency by ~60% in tight evaluation paths like intent classification.
**Action:** When calculating maximum frequencies or scores across multiple categories in an inner loop, use inline integer counters and manually update a `best_score`/`best_intent` tracker rather than building a full dictionary and passing it to built-in aggregate functions.

## 2023-10-27 - Inline vector dot product and norm accumulation
**Learning:** In Python, calculating vector math (like cosine similarity) using multiple generator expressions (e.g., `sum(x * y for...)` and `math.sqrt(sum(x * x for...))`) is inefficient. Replacing these with a single-pass inline `for` loop that accumulates the dot product and squared sums simultaneously significantly improves performance by reducing generator overhead and redundant iterations.
**Action:** When calculating similarity or distance manually between Python lists, write a single `for x, y in zip(a, b):` loop rather than using multiple `sum()` calls.

## 2023-10-27 - Fast reverse iteration for sorted sliding windows
**Learning:** Calculating counts over a chronologically sorted collection (like a sliding window rate limiter in `agent_gantry/core/rate_limiter.py`) using generator expressions that iterate over the entire collection (e.g., `sum(1 for t in history if t >= threshold)`) is highly inefficient for large windows.
**Action:** Use a reverse iterator (`for t in reversed(history):`) and break early when `t < threshold` to reduce O(N) operations to O(1).

## 2026-05-18 - Fast reverse iteration for sorted sliding windows in stats
**Learning:** Similar to sliding window limit checking, generating stats over a chronologically sorted collection (like the recent calls metric in `agent_gantry/core/rate_limiter.py`'s `get_stats`) using a list comprehension over the entire collection (`len([t for t in history if now - t < 60])`) is highly inefficient for large histories.
**Action:** Use a reverse iterator (`for t in reversed(history):`) and break early when the time window condition is no longer met, converting an O(N) operation to O(1).
## 2026-06-06 - Inline loop for vector norm calculation
**Learning:** In Python, calculating a vector norm using a generator expression inside `math.sqrt(sum(x * x for x in vec))` incurs unnecessary generator overhead. Replacing the generator expression with a single-pass inline `for` loop `for x in vec: norm_sq += x * x` significantly improves performance by reducing overhead, similar to optimizing cosine similarity calculations.
**Action:** When calculating norms or sums over lists in performance-critical paths, use inline `for` loops instead of generator expressions.

## 2026-06-12 - Optimize generator expressions in sum() for counting
**Learning:** In Python, using generator expressions inside sum() for counting items (e.g., sum(1 for x in iterable if condition)) incurs significant generator overhead. For performance-critical paths, replacing these with an inline for loop that manually accumulates a counter variable provides a measurable speedup.
**Action:** Replace generator expressions used inside sum() for counting with an inline for loop.
## 2026-06-23 - Optimize sliding window histories
**Learning:** Using `list.pop(0)` for rolling histories incurs O(N) overhead because all subsequent elements must be shifted in memory. This creates a bottleneck in tight loops or long-running instances.
**Action:** Use `collections.deque(maxlen=N)` for O(1) appending and automatic truncation from either end.
## 2026-06-25 - Push LanceDB operations to database engine
**Learning:** When querying LanceDB tables in Python, loading the entire table into memory via `table.to_arrow().to_pylist()` for filtering, counting, or pagination creates an O(N) memory bottleneck on large datasets. Additionally, retrieving all columns when only specific ones are needed (like 'id' and 'fingerprint') increases the payload significantly.
**Action:** Use LanceDB native query builders like `search().where(condition).limit(limit).offset(offset).to_list()`, `count_rows(condition)`, and `.select(["col1", "col2"])` instead of pulling the data into Python lists to process them.

## 2026-08-19 - Avoid row-wise dictionary allocation in PyArrow
**Learning:** When retrieving specific columns from LanceDB/PyArrow tables to build a Python dictionary mapping, calling `.to_pylist()` on the entire table allocates a dictionary for every row, causing a memory and CPU bottleneck for large datasets.
**Action:** Use columnar extraction via `.select(['col1', 'col2']).to_arrow()` and directly construct the mapping by zipping the specific column lists (e.g., `dict(zip(table['col1'].to_pylist(), table['col2'].to_pylist()))`).

## 2026-08-25 - Avoid row-wise dictionary allocation with LanceDB to_list for single columns
**Learning:** When retrieving a single column (like JSON strings) from LanceDB, using `.to_list()` allocates a dictionary for every row just to wrap the single field, causing O(N) memory overhead and slower execution on large tables.
**Action:** Use columnar extraction via `.select(['col']).to_arrow()` and then extract the list of values directly using `table['col'].to_pylist()`. This avoids dictionary allocation per row.
## 2026-09-03 - Optimize generator expressions in sum() for counting
**Learning:** In Python, using generator expressions inside sum() for counting items (e.g., sum(1 for x in iterable if condition)) incurs significant generator overhead. For performance-critical paths, replacing these with an inline for loop that manually accumulates a counter variable provides a measurable speedup.
**Action:** Replace generator expressions used inside sum() for counting with an inline for loop.
