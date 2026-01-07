# **High-Performance Algorithmic Trading Architecture: Optimizing Rust-Python Workflows for ARM Silicon and LLM-Driven Strategy Evaluation**

## **1\. Executive Summary**

The discipline of algorithmic trading is currently navigating a pivotal transition in its technological foundation, driven by the convergence of three distinct innovations: the proliferation of high-efficiency ARM-based silicon (exemplified by Apple’s M-series and AWS Graviton), the maturation of systems programming languages like Rust within the Python ecosystem, and the advent of Large Language Models (LLMs) capable of complex semantic reasoning. For the independent quantitative trader or boutique firm operating within the hardware constraints of 8GB to 24GB of RAM, these advancements necessitate a departure from traditional, memory-intensive Python-only architectures. The legacy stack—typically reliant on Pandas for data handling and purely Pythonic event loops for simulation—is increasingly ill-suited for the era of high-frequency tick data and iterative strategy optimization, struggling to maximize the parallel processing capabilities of modern Unified Memory Architectures (UMA).

This report presents a comprehensive architectural blueprint for a "Modern Tech Stack" specifically optimized for Apple Silicon and ARM VPS environments. The core of this architecture rests on a hybrid paradigm: utilizing **NautilusTrader**, a Rust-based, event-driven engine, to handle the computationally intensive task of market simulation with deterministic latency; leveraging **Polars** and **Apache Arrow** for zero-copy data manipulation that circumvents the memory overhead of standard DataFrames; and integrating **VectorBT** for rapid, vectorized post-trade analysis. Crucially, the architecture replaces the traditional, resource-heavy local machine learning model with an **API-based LLM Feedback Loop**. By offloading the qualitative evaluation of backtest logs to cloud-based models like Claude 3.5 Sonnet or GPT-4o, traders can implement an "Agentic Optimization" workflow that diagnoses strategy failures through semantic analysis rather than brute-force parameter sweeping, all without taxing the local machine's limited RAM. This report details the selection rationale, technical implementation, and strategic advantages of this stack, demonstrating how it enables institutional-grade research capabilities on consumer-grade hardware.

## **2\. The Hardware Paradigm: ARM Silicon and Quantitative Infrastructure**

To understand the necessity of a Rust-based stack, one must first analyze the underlying hardware shift that has occurred with the widespread adoption of ARM64 architectures, particularly Apple’s Silicon (M1/M2/M3) and server-side equivalents. Traditional quantitative software was largely optimized for x86\_64 architectures with discrete memory spaces for CPU and GPU. The shift to ARM introduces the Unified Memory Architecture (UMA), which fundamentally changes how data should be handled for maximum performance.

### **2.1 Unified Memory and Zero-Copy Implications**

The Apple M3 chip, a representative of modern ARM performance, utilizes UMA, granting both the CPU and GPU access to a single pool of high-bandwidth memory. In a trading context, this is transformative. In traditional architectures, moving data between a simulation process and an analysis process might involve costly serialization and memory copying, consuming valuable RAM and CPU cycles. On M3 systems, however, efficiency is maximized by keeping data in a consistent, contiguous memory format that can be accessed by multiple processes or threads without duplication.1

This hardware reality makes **Apache Arrow** the indispensable standard for the data layer. Arrow provides a language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware. When a backtesting engine is built on Rust (which can interact directly with C-compatible memory layouts) and utilizes Arrow, it can perform "zero-copy" reads. This means that a dataset loaded into memory by Polars (a Rust-based DataFrame library) can be read by NautilusTrader (a Rust-based engine) effectively by passing a pointer, rather than duplicating the gigabytes of tick data.3 For a user operating with 8GB or 16GB of RAM, this distinction is binary: it is the difference between being able to backtest a year of tick data or crashing with an Out-of-Memory (OOM) error.

### **2.2 The Role of Rust in Low-Latency Simulation**

The preference for Rust over C++ or pure Python in this modern stack is not merely syntactic but structural. Rust’s ownership model guarantees memory safety without a garbage collector, a critical feature for backtesting engines where non-deterministic pauses (garbage collection spikes common in Java or Python) can invalidate latency simulations. On ARM architectures, the Rust compiler (rustc) is exceptionally mature, capable of generating highly optimized aarch64 machine code that leverages the RISC (Reduced Instruction Set Computer) nature of the processor.2

Snippets indicate that the Python Global Interpreter Lock (GIL) remains the primary bottleneck for parallelization in pure Python backtesters. While Python threads cannot execute bytecode simultaneously, Rust threads can. By pushing the "hot path"—the event loop, order matching logic, and risk checks—into a compiled Rust binary that binds to Python, the system unlocks true parallelism.3 This allows the backtester to utilize the high core count of M3 chips (often 8, 10, or 12 cores) to process multiple strategy instances or data feeds concurrently, a feat impossible for standard Pandas-based loops.

## ---

**3\. The Core Engine: Comparative Analysis and Selection**

The selection of the core engine dictates the fidelity of the simulation and the developer experience. The requirement is a "Rust engine \+ Python" workflow, which filters out pure Rust frameworks like Barter-rs 5 requiring strategy logic in Rust, and pure Python frameworks like Backtrader which lack the requisite speed. The analysis narrows down to two primary candidates: **NautilusTrader** and **hftbacktest**.

### **3.1 NautilusTrader: The Generalist Powerhouse**

**NautilusTrader** emerges as the superior general-purpose choice for the "experienced trader" persona described in the query. It is a production-grade, event-driven algorithmic trading platform where the core components are written in Rust, yet it exposes a comprehensive, idiomatic Python API via PyO3.6

* **Architecture & Performance:** NautilusTrader uses an actor-based concurrency model, allowing different components (data ingestion, strategy logic, risk management, execution) to operate asynchronously. This aligns perfectly with the event-driven nature of live markets. Because the core data structures (OrderBook, QuoteTick, Bar) are implemented as Rust structs, they are extremely memory efficient compared to Python objects. A Python object carries significant overhead (often 28 bytes \+ payload), whereas a Rust struct can be packed tightly. When simulating millions of ticks, this efficiency effectively multiplies the available RAM.8  
* **Parity and Production:** A distinct advantage of NautilusTrader is its focus on "backtest-live parity." The engine used for backtesting is the exact same binary used for live trading; only the adapters change (from a DataCatalog to a WebSocket adapter). This eliminates the "implementation risk" where a strategy behaves differently in production than in simulation due to discrepancies in code paths.6  
* **ARM Compatibility:** The project publishes pre-built binary wheels for macOS ARM64 (macosx\_14\_0\_arm64), ensuring that installation is a simple pip install operation without requiring complex compilation toolchains or Rosetta translation layers, which would degrade performance.10

### **3.2 hftbacktest: The Microstructure Specialist**

**hftbacktest** represents a more specialized tool, designed specifically for High-Frequency Trading (HFT) and market-making strategies where queue position and latency modeling are paramount.11

* **Technology:** Unlike NautilusTrader’s actor model, hftbacktest relies heavily on **Numba**, a JIT compiler for Python that translates a subset of Python and NumPy code into fast machine code. Recent versions have begun rewriting core components in Rust to handle Level 3 (Market-by-Order) data reconstruction and latency simulation more efficiently.11  
* **Use Case:** This tool excels at reconstructing full order books from incremental feeds (L2/L3) and simulating the probability of a limit order being filled based on its queue position. It is the correct choice if the user’s strategy is a market-making algorithm dependent on capturing the spread.  
* **Limitations:** The reliance on Numba imposes constraints on the Python code a user can write within the strategy loop (e.g., limited support for third-party libraries or standard Python objects inside JIT-compiled functions). It lacks the broad ecosystem of exchange adapters and portfolio management features found in NautilusTrader.12

### **3.3 Selection Verdict**

For a general-purpose "High-Performance" stack that balances speed, flexibility, and ease of use, **NautilusTrader** is the recommended core. Its native support for Parquet catalogs and its PyO3 bindings offer the best integration point for the subsequent layers of the stack (Polars and LLMs). hftbacktest should be reserved for specialized microstructure research.

## ---

**4\. Data Engineering: The Zero-Copy Pipeline**

The traditional data science workflow involving Pandas and CSV files is the primary cause of memory exhaustion on constrained hardware. Loading a 2GB CSV file into Pandas often creates a DataFrame occupying 6-10GB of RAM due to inefficient string handling and object overhead. The modern stack mitigates this through **Polars** and **Parquet**.

### **4.1 Polars: The Engine for Data Manipulation**

**Polars** is a DataFrame library written in Rust, built specifically on the Apache Arrow memory model.4 It is the critical enabler for analyzing large datasets on an 8GB-24GB machine.

* **Lazy Execution:** Polars distinguishes between "eager" execution (doing work immediately) and "lazy" execution (building a query plan). In a memory-constrained environment, LazyFrames allow the user to define complex transformations (e.g., calculating rolling volatility on 10 years of tick data) without loading the entire dataset. Polars optimizes the query plan and streams the data in chunks, processing the result row-by-row or batch-by-batch. This is known as "out-of-core" processing and effectively decouples dataset size from RAM limitations.4  
* **Performance on ARM:** Polars is multithreaded. It automatically partitions workloads across the available cores of the M3 chip. Benchmarks consistently show Polars outperforming Pandas by factors of 10-100x on aggregation and join tasks, while using a fraction of the memory.15  
* **Integration with Nautilus:** NautilusTrader's configuration objects and data catalogs are designed to interface with Arrow. By using Polars to clean and prepare data, the trader ensures the data is already in the optimal format for the backtesting engine.

### **4.2 Storage Strategy: Parquet vs. ArcticDB**

The user specifically queried regarding ArcticDB. **ArcticDB** is a high-performance, serverless DataFrame database that excels at handling time-series data, offering features like compression-on-write and versioning (time-travel).17

* **Analysis:** ArcticDB is highly effective for teams needing to version-control their data or handle massive tick repositories distributed across S3. However, for a single-user setup on a VPS, it introduces architectural complexity (managing the LMDB or S3 backend).  
* **The Parquet Alternative:** A simpler, equally performant approach for this persona is a **Partitioned Parquet Data Lake**. By organizing data into a directory structure (e.g., data/symbol/year/month.parquet), the trader leverages the filesystem as the database. Parquet files are columnar, highly compressed (using Snappy or Zstd), and support "predicate pushdown"—meaning the reader (Polars or Nautilus) can skip reading entire chunks of the file if they don't match the query date range.4  
* **Recommendation:** Use a local Parquet directory structure managed by Polars. This avoids the overhead of a database service while retaining the performance benefits of columnar storage. It aligns perfectly with NautilusTrader’s ParquetDataCatalog, which is designed to stream specifically from this format.20

### **4.3 Data Ingestion Pattern**

The optimal workflow involves fetching raw data (via APIs like Databento or CCXT), immediately streaming it into a Polars LazyFrame, casting types to their most memory-efficient representations (e.g., Float32 instead of Float64 if precision allows, Categorical for symbols), and writing to Parquet. This ensures that the expensive "ETL" process never spikes RAM usage.4

## ---

**5\. Analytics & Metrics: The Vectorized Bridge**

Once a backtest is complete, the event-driven engine produces a linear log of events (orders, fills, cancellations). Analyzing this log to determine strategy efficacy requires a different set of tools. While the simulation is event-driven, the analysis should be vectorized for speed.

### **5.1 VectorBT Integration**

**VectorBT (VBT)** is a "next-generation" backtesting library that operates entirely on Pandas/NumPy objects, accelerated by Numba.21 While it can run its own simulations, its Portfolio module is exceptionally powerful when used as an analytics engine for external trade lists.

* **The "Analyzer" Pattern:** Instead of using VectorBT to simulate the strategy, the architecture uses it to analyze the output of NautilusTrader. Nautilus exports the trade log (a Polars/Pandas DataFrame). This log is ingested into VectorBT using vbt.Portfolio.from\_orders() or vbt.Portfolio.from\_trades().22  
* **Capabilities:** Once the data is in a VBT Portfolio object, the user gains access to hundreds of vectorized performance metrics (Sharpe, Sortino, Calmar, Omega, Tail Ratio) calculated in milliseconds. Furthermore, VectorBT enables the rapid simulation of "what-if" scenarios on the realized trades—for example, "What would my Sharpe Ratio be if I had applied a 5% trailing stop to these exact entries?" This allows for post-hoc optimization without re-running the computationally expensive event-driven simulation.24  
* **Memory Efficiency:** VectorBT is highly optimized for memory. By operating on NumPy arrays and utilizing broadcasting, it minimizes the memory footprint of the analysis phase, fitting comfortably within the available RAM even for large trade lists.25

### **5.2 Visualization and Reporting**

VectorBT integrates deeply with Plotly, allowing for the generation of interactive, high-density performance dashboards (equity curves, underwater plots, trade heatmaps). These visualizations can be exported as HTML or JSON, serving as the input for the next stage of the pipeline: the LLM Agent.26

## ---

**6\. The LLM-Driven Optimization Loop: "Agentic" Strategy Tuning**

The most forward-looking component of this architecture is the integration of Large Language Models to automate the interpretation of backtest results. Traditionally, a trader manually inspects charts to diagnose why a strategy failed. This process is slow and prone to cognitive bias. The "Modern Tech Stack" replaces this with an automated loop where an LLM acts as a "Senior Risk Manager."

### **6.1 The "LLM as a Judge" Workflow**

The concept of "LLM as a Judge," often used to evaluate chatbot performance, is adapted here for financial diagnostics.27 The LLM is not asked to predict price action (a task for which they are ill-suited); rather, it is asked to **reason about the relationship between strategy parameters and performance outcomes**.

* **Workflow:**  
  1. **Metric Aggregation:** Python scripts aggregate the Nautilus/VectorBT outputs into a condensed JSON object. This includes high-level metrics (Win Rate, Drawdown Duration) and, crucially, a sample of the "Worst 5 Trades" and "Best 5 Trades" with their associated market context (volatility regime, time of day).  
  2. **Semantic Analysis:** This JSON is sent to the LLM API (e.g., Claude 3.5 Sonnet). A specialized System Prompt instructs the model to analyze the failure modes. For instance, the LLM might deduce: *"The strategy consistently incurs losses during the opening 15 minutes of the NYSE session, suggesting high sensitivity to opening volatility. Recommendation: Add a time-filter to suppress entries before 09:45 AM."*  
  3. **Structured Feedback:** The LLM is constrained to output its recommendations in a valid JSON format, detailing specific parameter updates (e.g., {"time\_filter\_start": "09:45"}).  
  4. **Automated Iteration:** The Python controller parses this JSON, updates the strategy configuration, and triggers a new NautilusTrader backtest run.

### **6.2 Why API-Based?**

The user requirement specifies 8GB-24GB RAM. Running a local LLM with sufficient reasoning capability (e.g., Llama-3-70B) is impossible on this hardware; it would require significantly more VRAM/RAM. Smaller local models (7B parameters) often lack the nuance required for complex financial reasoning and struggle with strict JSON adherence. API-based models (Claude/GPT) offer state-of-the-art reasoning with zero local memory footprint, effectively "cloud-bursting" the most intelligent part of the stack.29

### **6.3 Prompt Engineering for Financial Reasoning**

To make this effective, the prompt must be engineered to provide context and enforce structure.31

* **Role Definition:** "You are an expert Quantitative Researcher and Risk Manager."  
* **Context Injection:** The prompt must include a "Data Dictionary" explaining what the parameters control.  
* **Chain of Thought:** The prompt should encourage the LLM to "think step-by-step" about the correlation between market regimes (e.g., trending vs. ranging) and the strategy's performance before outputting the JSON configuration. This significantly reduces "hallucinated" optimizations.31

## ---

**7\. Implementation Architecture: The "Modern Tech Stack"**

The following section synthesizes the component analysis into a cohesive, actionable architecture. This "Rust-Quant-LLM" stack is designed to be modular, efficient, and robust.

### **7.1 Architecture Diagram**

1. **Data Ingestion Layer (Python/Polars):**  
   * **Source:** Exchange APIs (CCXT, Databento) fetch raw tick/bar data.  
   * **Processing:** Polars cleans, types, and sorts data.  
   * **Storage:** Partitioned Parquet files on local SSD (e.g., data/btc/2024.parquet).  
2. **Simulation Layer (Rust/NautilusTrader):**  
   * **Engine:** BacktestNode instantiated in Python.  
   * **Input:** ParquetDataCatalog streams data from the storage layer.  
   * **Logic:** Strategy class (Python) defines entry/exit logic.  
   * **Execution:** Rust core processes events and matches orders.  
3. **Analytics Layer (Python/VectorBT):**  
   * **Ingest:** Order/Fill logs exported from Nautilus to Polars DataFrame.  
   * **Processing:** vbt.Portfolio.from\_orders() generates equity curve and metrics.  
   * **Output:** JSON summary of performance statistics.  
4. **Optimization Layer (Cloud LLM API):**  
   * **Input:** JSON summary sent to Claude/GPT API.  
   * **Reasoning:** LLM analyzes logs and suggests parameter changes.  
   * **Feedback:** JSON response updates the Simulation Layer configuration.

### **7.2 Implementation Details & Code Patterns**

**Step 1: Data Preparation with Polars**

Python

import polars as pl

def prepare\_data(csv\_path, output\_path):  
    \# Lazy evaluation allows processing files larger than RAM  
    q \= (  
        pl.scan\_csv(csv\_path)  
       .with\_columns(\[  
            pl.col("timestamp").str.to\_datetime(),  
            pl.col("price").cast(pl.Float64),  
            pl.col("volume").cast(pl.Float64)  
        \])  
       .sort("timestamp")  
    )  
    \# Sink to Parquet (streaming execution)  
    q.sink\_parquet(output\_path, compression="zstd")

**Step 2: NautilusTrader Simulation**

Python

from nautilus\_trader.backtest.node import BacktestNode, BacktestRunConfig  
from nautilus\_trader.persistence.catalog import ParquetDataCatalog

def run\_backtest(params):  
    catalog \= ParquetDataCatalog("data\_catalog")  
    node \= BacktestNode(  
        config=BacktestRunConfig(  
            data=,  
            strategies=,  
        )  
    )  
    results \= node.run()  
    return node.trader.generate\_order\_fills\_report() \# Returns Pandas/Polars DF

**Step 3: VectorBT Analysis**

Python

import vectorbt as vbt

def analyze\_results(fills\_df):  
    \# Convert Nautilus fills to VBT Portfolio  
    \# VectorBT expects an index of timestamps and columns of assets  
    pf \= vbt.Portfolio.from\_orders(  
        close=fills\_df\['price'\],  \# Approximate with fill price or join with OHLC  
        size=fills\_df\['quantity'\],  
        direction=fills\_df\['side'\], \# Map BUY/SELL to direction  
        freq='1m'  
    )  
      
    metrics \= {  
        "sharpe": pf.sharpe\_ratio(),  
        "max\_dd": pf.max\_drawdown(),  
        "win\_rate": pf.trades.win\_rate(),  
        "worst\_drawdown\_duration": pf.drawdowns.max\_duration()  
    }  
    return metrics

**Step 4: LLM Optimization Loop**

Python

import anthropic  
import json

client \= anthropic.Anthropic(api\_key="your\_key")

def optimize\_strategy(current\_params, metrics\_json):  
    prompt \= f"""  
    Analyze the following backtest metrics: {json.dumps(metrics\_json)}.  
    Current parameters: {json.dumps(current\_params)}.  
    Suggest improvements to reduce Max Drawdown.  
    Output JSON format: {{ "reasoning": "...", "new\_params": {{...}} }}  
    """  
      
    response \= client.messages.create(  
        model="claude-3-5-sonnet-20240620",  
        max\_tokens=1000,  
        messages=\[{"role": "user", "content": prompt}\]  
    )  
      
    suggestion \= json.loads(response.content.text)  
    return suggestion\['new\_params'\]

## **8\. Strategic Risks and Recommendations**

### **8.1 Mitigating "Hallucinated Overfitting"**

A primary risk of using LLMs for optimization is that they may suggest parameters that "fit" the noise in the log rather than the signal (e.g., "avoid trading on Tuesdays"). To mitigate this, the prompt should explicitly request *hypothesis-driven* changes grounded in market logic (e.g., "reduce position size during high volatility") rather than arbitrary curve fitting. Additionally, walk-forward validation (testing the LLM's suggested parameters on unseen data) is mandatory.33

### **8.2 Data Privacy**

While trade logs (entry/exit timestamps and prices) are generally not considered Personally Identifiable Information (PII) or highly sensitive intellectual property in isolation, traders should be cautious. Utilizing Enterprise API tiers (which often guarantee zero data retention for training) is recommended over consumer tiers. Anonymizing symbols (e.g., renaming "BTCUSDT" to "Asset\_A") in the logs sent to the LLM can further protect proprietary alpha.34

### **8.3 Conclusion**

The transition to ARM-based hardware necessitates a modernization of the quantitative stack. By adopting **NautilusTrader** for execution, **Polars/Parquet** for data engineering, and **VectorBT** for analysis, a trader can achieve high-performance backtesting on limited RAM. Augmenting this with an **LLM API** optimization loop creates a powerful, semi-autonomous research assistant that leverages the best of local efficiency and cloud intelligence. This architecture is not merely a workaround for hardware limits; it is a sophisticated, forward-looking standard for independent quantitative finance.

#### **Works cited**

1. oxideai/mlx-rs: Unofficial Rust bindings to Apple's mlx framework, accessed January 6, 2026, [https://github.com/oxideai/mlx-rs](https://github.com/oxideai/mlx-rs)  
2. With Rust on Apple Silicone; any gotchas to watch out for? \- Reddit, accessed January 6, 2026, [https://www.reddit.com/r/rust/comments/1h43y1t/with\_rust\_on\_apple\_silicone\_any\_gotchas\_to\_watch/](https://www.reddit.com/r/rust/comments/1h43y1t/with_rust_on_apple_silicone_any_gotchas_to_watch/)  
3. More Than Python in Backtesting (Part 2\) | Balaena Quant Insights, accessed January 6, 2026, [https://medium.com/balaena-quant-insights/more-than-python-in-backtesting-part-2-b0520eaaace9](https://medium.com/balaena-quant-insights/more-than-python-in-backtesting-part-2-b0520eaaace9)  
4. Parquet \- Polars user guide, accessed January 6, 2026, [https://docs.pola.rs/user-guide/io/parquet/](https://docs.pola.rs/user-guide/io/parquet/)  
5. barter \- Rust \- Docs.rs, accessed January 6, 2026, [https://docs.rs/barter](https://docs.rs/barter)  
6. nautilus\_backtest \- Rust \- Docs.rs, accessed January 6, 2026, [https://docs.rs/nautilus-backtest](https://docs.rs/nautilus-backtest)  
7. NautilusTrader, accessed January 6, 2026, [https://nautilustrader.io/](https://nautilustrader.io/)  
8. Backtesting | NautilusTrader Documentation, accessed January 6, 2026, [https://nautilustrader.io/docs/latest/concepts/backtesting/](https://nautilustrader.io/docs/latest/concepts/backtesting/)  
9. High Performance Backtesting and Trading with NautilusTrader Part 2, accessed January 6, 2026, [https://www.youtube.com/watch?v=Be0uqb8K3W4](https://www.youtube.com/watch?v=Be0uqb8K3W4)  
10. nautilus\_trader \- PyPI, accessed January 6, 2026, [https://pypi.org/project/nautilus\_trader/1.208.0/](https://pypi.org/project/nautilus_trader/1.208.0/)  
11. High-Frequency Trading Backtesting Tool, accessed January 6, 2026, [https://hftbacktest.readthedocs.io/en/py-v2.3.0/](https://hftbacktest.readthedocs.io/en/py-v2.3.0/)  
12. HftBacktest — hftbacktest, accessed January 6, 2026, [https://hftbacktest.readthedocs.io/](https://hftbacktest.readthedocs.io/)  
13. Open-Sourcing High-Frequency Trading and Market-Making ..., accessed January 6, 2026, [https://www.reddit.com/r/quant/comments/1dhz9m9/opensourcing\_highfrequency\_trading\_and/](https://www.reddit.com/r/quant/comments/1dhz9m9/opensourcing_highfrequency_trading_and/)  
14. Comparison with other tools \- Polars user guide, accessed January 6, 2026, [https://docs.pola.rs/user-guide/misc/comparison/](https://docs.pola.rs/user-guide/misc/comparison/)  
15. Updated PDS-H benchmark results (May 2025\) \- Polars, accessed January 6, 2026, [https://pola.rs/posts/benchmarks/](https://pola.rs/posts/benchmarks/)  
16. The Great Data Processing Showdown: Pandas vs Polars vs Spark ..., accessed January 6, 2026, [https://medium.com/@sriram-narasim/the-great-data-processing-showdown-pandas-vs-polars-vs-spark-vs-duckdb-e37978c4dffd](https://medium.com/@sriram-narasim/the-great-data-processing-showdown-pandas-vs-polars-vs-spark-vs-duckdb-e37978c4dffd)  
17. 23 million rows of intra-day FX tick data in Excel using ArcticDB and ..., accessed January 6, 2026, [https://www.pyxll.com/blog/arcticdb/](https://www.pyxll.com/blog/arcticdb/)  
18. Pandas Can't Handle This: How ArcticDB Powers Massive Datasets, accessed January 6, 2026, [https://towardsdatascience.com/pandas-cant-handle-this-how-arcticdb-powers-massive-datasets/](https://towardsdatascience.com/pandas-cant-handle-this-how-arcticdb-powers-massive-datasets/)  
19. Working With Apache Parquet for Faster Data Processing \- Daft, accessed January 6, 2026, [https://www.daft.ai/blog/working-with-the-apache-parquet](https://www.daft.ai/blog/working-with-the-apache-parquet)  
20. Persistence | NautilusTrader Documentation, accessed January 6, 2026, [https://nautilustrader.io/docs/latest/api\_reference/persistence/](https://nautilustrader.io/docs/latest/api_reference/persistence/)  
21. vectorbt: Getting started, accessed January 6, 2026, [https://vectorbt.dev/](https://vectorbt.dev/)  
22. base \- vectorbt, accessed January 6, 2026, [https://vectorbt.dev/api/portfolio/base/](https://vectorbt.dev/api/portfolio/base/)  
23. trades \- vectorbt, accessed January 6, 2026, [https://vectorbt.dev/api/portfolio/trades/](https://vectorbt.dev/api/portfolio/trades/)  
24. Backtesting with VectorBT: A Beginner's Guide | by Trading Dude, accessed January 6, 2026, [https://medium.com/@trading.dude/backtesting-with-vectorbt-a-beginners-guide-8b9c0e6a0167](https://medium.com/@trading.dude/backtesting-with-vectorbt-a-beginners-guide-8b9c0e6a0167)  
25. Performance \- VectorBT® PRO, accessed January 6, 2026, [https://vectorbt.pro/features/performance/](https://vectorbt.pro/features/performance/)  
26. Features \- vectorbt, accessed January 6, 2026, [https://vectorbt.dev/getting-started/features/](https://vectorbt.dev/getting-started/features/)  
27. LLM-as-a-judge: can AI systems evaluate human responses and ..., accessed January 6, 2026, [https://toloka.ai/blog/llm-as-a-judge-can-ai-systems-evaluate-model-outputs/](https://toloka.ai/blog/llm-as-a-judge-can-ai-systems-evaluate-model-outputs/)  
28. LLM-as-a-judge: a complete guide to using LLMs for evaluations, accessed January 6, 2026, [https://www.evidentlyai.com/llm-guide/llm-as-a-judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)  
29. Claude code 100k options trading : r/ClaudeCode \- Reddit, accessed January 6, 2026, [https://www.reddit.com/r/ClaudeCode/comments/1ps8rew/claude\_code\_100k\_options\_trading/](https://www.reddit.com/r/ClaudeCode/comments/1ps8rew/claude_code_100k_options_trading/)  
30. MCP Server-Bridge Your AI Agents to QuantConnect, accessed January 6, 2026, [https://www.quantconnect.com/mcp](https://www.quantconnect.com/mcp)  
31. Prompting Strategies for Financial Analysis \- Claude Help Center, accessed January 6, 2026, [https://support.claude.com/en/articles/12220277-prompting-strategies-for-financial-analysis](https://support.claude.com/en/articles/12220277-prompting-strategies-for-financial-analysis)  
32. Claude AI Prompting Techniques: structure, examples, and best ..., accessed January 6, 2026, [https://www.datastudios.org/post/claude-ai-prompting-techniques-structure-examples-and-best-practices](https://www.datastudios.org/post/claude-ai-prompting-techniques-structure-examples-and-best-practices)  
33. Hyperparameters: Optimization and Tuning for Machine Learning, accessed January 6, 2026, [https://blog.quantinsti.com/hyperparameter/](https://blog.quantinsti.com/hyperparameter/)  
34. How to Use LLMs for Log File Analysis: Examples, Workflows, and ..., accessed January 6, 2026, [https://www.splunk.com/en\_us/blog/learn/log-file-analysis-llms.html](https://www.splunk.com/en_us/blog/learn/log-file-analysis-llms.html)