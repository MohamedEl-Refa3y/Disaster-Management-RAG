# 🌪️ Disaster Management AI System — RAG-Based Intelligence using Jina Embeddings & ALLAM 7B

## 🧭 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** workflow designed for **a prototype Disaster Management in Hail**, focusing on improving response intelligence, data accessibility, and decision-making capabilities during emergency situations.

The system integrates **structured disaster data** from SQL Server with **semantic embeddings** and a **Large Language Model (LLM)** — enabling real-time question answering and situational awareness through retrieval-augmented reasoning.

---

## 🧠 System Workflow

### 1. Data Ingestion
All relevant disaster-related data (such as emergency logs, resource availability, and incident reports) are collected and stored in a **SQL Server database**.  
- Each record represents an event, alert, or report entry.  
- Metadata such as timestamps, severity levels, and region identifiers are included.

A **metadata.json** file is automatically generated from SQL Server tables to provide structured context for the RAG pipeline.

---

### 2. Embedding Generation
We utilize **Jina Embeddings v3**, a high-performance multilingual embedding model that transforms text into dense vector representations.  
These embeddings capture semantic meaning, enabling similarity-based retrieval.

**Key roles of Jina Embeddings:**
- Convert disaster metadata and reports into vectors.  
- Enable semantic search and contextual understanding.  
- Improve retrieval accuracy even when queries use different terminology.

---

### 3. Vector Storage with FAISS
All generated embeddings are stored in a **FAISS (Facebook AI Similarity Search)** index for efficient retrieval.  

**Why FAISS:**
- Scalable to thousands of disaster records.  
- Fast similarity search using cosine distance or inner product.  
- Supports vector compression for optimized performance.

---

### 4. Query Understanding (User Input)
When a user or operator inputs a query — for example:  
> “List the most recent flood alerts in the northern Hail region.”

The system:
1. Converts the query into an embedding using **Jina v3**.  
2. Searches FAISS for the top similar entries.  
3. Retrieves relevant metadata and disaster details from the SQL dataset.

---

### 5. Context Augmentation & Reasoning (RAG Stage)
The retrieved information is passed along with the user’s query to the **ALLAM 7B** model.  
This model performs **contextual reasoning** and generates coherent responses or summaries.

**LLM Responsibilities:**
- Fuse factual context (from FAISS) with the input query.  
- Generate structured and human-like explanations.  
- Provide actionable insights for disaster response.

---

### 6. Deployment and Web App
The complete pipeline was deployed using **FastAPI**, creating a **lightweight local web application** for interactive usage.

**Key Deployment Steps:**
- Designed RESTful endpoints in FastAPI for querying the RAG pipeline.  
- Integrated FAISS, metadata lookup, and LLM generation through API routes.  
- Built a simple **web-based interface (local host)** allowing users to:
  - Input disaster-related queries.  
  - View AI-generated responses and summaries.  
  - Monitor disaster status updates interactively.

The app runs locally, providing a user-friendly interface for emergency operators and analysts without requiring cloud infrastructure.

---

### 7. Output Generation
The final output is an **AI-generated summary or analytical report**, combining:
- Retrieved factual data (from FAISS and SQL metadata).  
- Reasoned explanations (via ALLAM 7B).  
- Context-aware recommendations.

Example response:
> “In northern Hail, three flood alerts were detected within the last 24 hours. Emergency response units A and B have been dispatched. Risk level elevated to Tier 3 based on rainfall intensity.”

---

## 🧩 Tech Stack Summary

| Component | Technology | Purpose |
|------------|-------------|----------|
| **Database** | Microsoft SQL Server | Structured disaster data storage |
| **Embeddings** | Jina Embeddings v3 | Semantic text vectorization |
| **Vector Database** | FAISS | Fast similarity-based retrieval |
| **Language Model** | ALLAM 7B | Contextual reasoning and response generation |
| **API Framework** | FastAPI | Deployment and local web interface |
| **Integration Logic** | Python | Data flow, vector indexing, and LLM orchestration |
| **Metadata File** | metadata.json | Structured reference for disaster context |

---

## 🌟 Features
- 🧠 **Context-Aware Disaster Intelligence** through RAG.  
- ⚙️ **Fast Retrieval** with FAISS indexing.  
- 💬 **Interactive Web App** built with FastAPI for local deployment.  
- 🗂️ **Metadata Integration** for structured and traceable context.  
- 🔍 **Semantic Search** using Jina Embeddings v3.  

---

## 🚨 Use Cases
- Real-time disaster monitoring and response coordination.  
- Querying specific events (e.g., floods, earthquakes, storms).  
- Summarizing historical incident data.  
- Generating emergency reports for decision-making.  

---

## 🚀 Future Enhancements
- Integration with **geospatial visualization (GIS)** for map-based disaster tracking.  
- Streaming ingestion from **IoT sensors** and live weather APIs.  
- Fine-tuning **ALLAM 7B** for domain-specific terminology.  
- Hosting on **cloud services (Azure / AWS)** for multi-user scalability.

---

## 👨‍💻 Author
**Developer:** Mohamed Elrefaay  
📧 **mohamedelrefaai45@gmail.com**  
🌐 [GitHub Profile](https://github.com/MohamedEl-Refa3y)

---
