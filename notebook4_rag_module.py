"""
KnoAD-Net Implementation - Notebook 4: RAG Module
=================================================
This notebook implements the Retrieval-Augmented Generation module:
- Build clinical knowledge base
- Embed documents with sentence transformers
- Implement retrieval pipeline
- Generate diagnostic explanations

Estimated Time: 3-4 hours
GPU Required: Optional (for faster embedding)
"""

# ============================================================================
# SECTION 1: Setup & Imports
# ============================================================================
print("=" * 80)
print("KNOADNET - PHASE 4: RAG MODULE")
print("=" * 80)


import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from langchain_text_splitters import RecursiveCharacterTextSplitter




# Load config
PROJECT_ROOT = 'D:\Projects\AI-Projects\Alzimers'
sys.path.append(PROJECT_ROOT)
from utils import set_seed, load_config, get_device

config = load_config(f'{PROJECT_ROOT}/config.json')
set_seed(config['random_seed'])

device = get_device()
print(f"✓ Using device: {device}")

# ============================================================================
# SECTION 2: Build Clinical Knowledge Base
# ============================================================================
print("\n[1/5] Building Clinical Knowledge Base")
print("-" * 80)

# Create clinical knowledge documents
clinical_documents = [
    {
        'source': 'NIA-AA 2011 Criteria',
        'content': """
        The National Institute on Aging-Alzheimer's Association (NIA-AA) 2011 criteria 
        define Alzheimer's disease through biomarkers and clinical symptoms. Key diagnostic 
        criteria include: (1) Progressive memory impairment affecting daily activities, 
        (2) Evidence of beta-amyloid deposition via PET or CSF, (3) Neuronal injury markers 
        such as elevated tau or reduced hippocampal volume on MRI, (4) MMSE scores typically 
        below 24 in mild dementia. The criteria distinguish between preclinical AD, mild 
        cognitive impairment (MCI) due to AD, and dementia due to AD based on severity of 
        cognitive and functional impairment.
        """,
        'category': 'diagnostic_criteria'
    },
    {
        'source': 'NIA-AA 2018 Research Framework',
        'content': """
        The 2018 NIA-AA Research Framework updates diagnostic criteria using the ATN 
        classification system: A (amyloid), T (tau), N (neurodegeneration). This framework 
        emphasizes biological markers over clinical symptoms for research purposes. Key 
        biomarkers include: Amyloid PET or CSF Aβ42 for A, tau PET or CSF p-tau for T, 
        and MRI hippocampal volume or FDG-PET for N. The framework recognizes the 
        Alzheimer's continuum from cognitively normal to dementia stages.
        """,
        'category': 'diagnostic_criteria'
    },
    {
        'source': 'MMSE Interpretation Guidelines',
        'content': """
        The Mini-Mental State Examination (MMSE) is a 30-point cognitive screening test. 
        Score interpretation: 24-30 indicates normal cognition, 18-23 suggests mild 
        cognitive impairment, 10-17 indicates moderate cognitive impairment, and scores 
        below 10 suggest severe cognitive impairment. However, MMSE scores must be 
        interpreted considering education level, age, and cultural background. A decline 
        of 3+ points per year suggests progressive dementia.
        """,
        'category': 'clinical_assessment'
    },
    {
        'source': 'MoCA Assessment Guidelines',
        'content': """
        The Montreal Cognitive Assessment (MoCA) is a 30-point screening tool more sensitive 
        than MMSE for detecting mild cognitive impairment. Score interpretation: 26-30 is 
        normal, 18-25 suggests MCI, below 18 indicates possible dementia. MoCA assesses 
        multiple cognitive domains: visuospatial/executive function, naming, memory, 
        attention, language, abstraction, delayed recall, and orientation. Add 1 point 
        if education is 12 years or less.
        """,
        'category': 'clinical_assessment'
    },
    {
        'source': 'MRI Biomarkers for AD',
        'content': """
        Structural MRI reveals characteristic patterns in Alzheimer's disease. Key findings 
        include: (1) Hippocampal atrophy, typically >30% volume loss in AD compared to 
        healthy controls, (2) Temporal lobe atrophy, particularly medial temporal regions, 
        (3) Ventricular enlargement due to global brain volume loss, (4) Posterior cingulate 
        and precuneus atrophy in early stages. Volumetric analysis of hippocampus shows 
        strong correlation with memory performance and disease progression. Atrophy rates 
        of >5% annually predict conversion from MCI to AD.
        """,
        'category': 'neuroimaging'
    },
    {
        'source': 'AD Progression Stages',
        'content': """
        Alzheimer's disease typically progresses through distinct stages: (1) Preclinical AD: 
        biomarker positive but cognitively normal, may last 10-20 years, (2) Mild Cognitive 
        Impairment (MCI): noticeable memory problems but preserved daily function, MMSE 24-27, 
        10-15% annual conversion to dementia, (3) Mild AD dementia: MMSE 20-24, difficulties 
        with complex tasks, needs some assistance, (4) Moderate AD: MMSE 10-19, significant 
        memory loss, needs help with daily activities, (5) Severe AD: MMSE <10, cannot 
        communicate effectively, total dependence on caregivers.
        """,
        'category': 'disease_progression'
    },
    {
        'source': 'APOE4 Genetic Risk',
        'content': """
        The APOE ε4 allele is the strongest genetic risk factor for late-onset Alzheimer's 
        disease. Having one APOE4 allele increases AD risk 3-4 fold, while two copies 
        increase risk 8-12 fold. APOE4 carriers typically show earlier age of onset 
        (5-10 years earlier) and faster disease progression. However, APOE4 is neither 
        necessary nor sufficient for AD development. Approximately 40-65% of AD patients 
        carry at least one APOE4 allele, compared to 25% of the general population.
        """,
        'category': 'genetics'
    },
    {
        'source': 'Differential Diagnosis',
        'content': """
        Alzheimer's disease must be differentiated from other dementias and cognitive 
        disorders: (1) Vascular dementia: stepwise decline, focal neurological signs, 
        white matter changes on MRI, (2) Lewy body dementia: visual hallucinations, 
        parkinsonism, fluctuating cognition, (3) Frontotemporal dementia: behavioral 
        changes, language problems, frontal/temporal atrophy, (4) Depression: potentially 
        reversible pseudo-dementia, (5) Normal aging: slower decline, preserved function. 
        Mixed pathologies are common in elderly patients.
        """,
        'category': 'differential_diagnosis'
    },
    {
        'source': 'Treatment Approaches',
        'content': """
        Current FDA-approved treatments for Alzheimer's disease include: (1) Cholinesterase 
        inhibitors (donepezil, rivastigmine, galantamine): improve symptoms in mild-moderate 
        AD, (2) NMDA antagonist (memantine): used in moderate-severe AD, can be combined 
        with cholinesterase inhibitors, (3) Anti-amyloid antibodies (aducanumab, lecanemab): 
        recently approved, target amyloid plaques, show modest clinical benefit. 
        Non-pharmacological interventions include cognitive stimulation, physical exercise, 
        social engagement, and caregiver education. Early diagnosis enables better treatment 
        planning and patient/family education.
        """,
        'category': 'treatment'
    },
    {
        'source': 'Clinical Decision Support',
        'content': """
        Comprehensive Alzheimer's assessment requires integration of multiple data sources: 
        (1) Clinical history: gradual onset, progressive decline, family history, 
        (2) Cognitive testing: MMSE, MoCA, detailed neuropsychological battery, 
        (3) Neuroimaging: MRI for structural changes, PET for functional/molecular imaging, 
        (4) Laboratory tests: rule out reversible causes (B12, thyroid, syphilis), 
        (5) Functional assessment: activities of daily living, instrumental ADLs. 
        Multidisciplinary evaluation improves diagnostic accuracy and enables personalized 
        treatment planning.
        """,
        'category': 'clinical_decision_support'
    }
]

# Add more documents for robustness
additional_docs = [
    {
        'source': 'Risk Factors',
        'content': """Risk factors for Alzheimer's include age (doubles every 5 years after 65), 
        family history, APOE4 genotype, cardiovascular disease, diabetes, hypertension, 
        obesity, smoking, head trauma, low education, and social isolation.""",
        'category': 'risk_factors'
    },
    {
        'source': 'Protective Factors',
        'content': """Protective factors against Alzheimer's include higher education, 
        cognitive reserve, regular physical exercise, Mediterranean diet, social engagement, 
        cognitive training, good sleep quality, and cardiovascular health management.""",
        'category': 'protective_factors'
    }
]

clinical_documents.extend(additional_docs)

print(f"✓ Created {len(clinical_documents)} clinical documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

chunks = []
for doc in clinical_documents:
    doc_chunks = text_splitter.split_text(doc['content'])
    for chunk in doc_chunks:
        chunks.append({
            'text': chunk.strip(),
            'source': doc['source'],
            'category': doc['category']
        })

print(f"✓ Split into {len(chunks)} chunks")

# ============================================================================
# SECTION 3: Embed Documents with Sentence Transformers
# ============================================================================
print("\n[2/5] Embedding Documents")
print("-" * 80)

# Load embedding model
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"✓ Model loaded: all-MiniLM-L6-v2")

# Embed all chunks
print("\nEmbedding document chunks...")
chunk_texts = [chunk['text'] for chunk in chunks]
embeddings = embedding_model.encode(
    chunk_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"✓ Created embeddings with shape: {embeddings.shape}")

# ============================================================================
# SECTION 4: Create Vector Store with ChromaDB
# ============================================================================
print("\n[3/5] Creating Vector Store")
print("-" * 80)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory=f"{config['directories']['rag_knowledge']}/chroma_db",
    anonymized_telemetry=False
))

# Create or get collection
collection_name = "clinical_knowledge"
try:
    collection = chroma_client.get_collection(collection_name)
    print(f"✓ Loaded existing collection: {collection_name}")
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Clinical knowledge for Alzheimer's diagnosis"}
    )
    print(f"✓ Created new collection: {collection_name}")

# Add documents to collection
print("\nAdding documents to vector store...")
ids = [f"doc_{i}" for i in range(len(chunks))]
metadatas = [{'source': chunk['source'], 'category': chunk['category']} for chunk in chunks]

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=chunk_texts,
    metadatas=metadatas
)

print(f"✓ Added {len(chunks)} documents to vector store")

# ============================================================================
# SECTION 5: Implement Retrieval Pipeline
# ============================================================================
print("\n[4/5] Implementing Retrieval Pipeline")
print("-" * 80)

class RAGRetriever:
    """Retrieval-Augmented Generation Retriever"""
    
    def __init__(self, collection, embedding_model, top_k=5):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(self, query, top_k=None):
        """
        Retrieve relevant documents for a query
        
        Args:
            query: str or dict with prediction context
            top_k: number of documents to retrieve
        
        Returns:
            List of retrieved documents with metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        # Convert query to string if dict
        if isinstance(query, dict):
            query_text = self._build_query_from_prediction(query)
        else:
            query_text = query
        
        # Embed query
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Retrieve from vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'category': results['metadatas'][0][i]['category'],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def _build_query_from_prediction(self, pred_context):
        """Build query string from prediction context"""
        query_parts = []
        
        if 'predicted_label' in pred_context:
            query_parts.append(f"diagnosis {pred_context['predicted_label']}")
        
        if 'mmse' in pred_context:
            query_parts.append(f"MMSE score {pred_context['mmse']}")
        
        if 'moca' in pred_context:
            query_parts.append(f"MoCA score {pred_context['moca']}")
        
        if 'attention_regions' in pred_context:
            query_parts.append(f"brain regions {' '.join(pred_context['attention_regions'])}")
        
        query_parts.append("Alzheimer's disease diagnostic criteria")
        
        return " ".join(query_parts)
    
    def generate_explanation(self, prediction_context, retrieved_docs):
        """
        Generate diagnostic explanation based on prediction and retrieved knowledge
        
        Args:
            prediction_context: dict with model predictions
            retrieved_docs: list of retrieved documents
        
        Returns:
            Structured explanation dict
        """
        # Extract key information
        label = prediction_context.get('predicted_label', 'Unknown')
        confidence = prediction_context.get('confidence', 0.0)
        mmse = prediction_context.get('mmse', 'N/A')
        moca = prediction_context.get('moca', 'N/A')
        
        # Build explanation (rule-based for demo; use LLM in production)
        explanation = {
            'diagnosis': label,
            'confidence': confidence,
            'rationale': [],
            'supporting_evidence': [],
            'clinical_guidelines': [],
            'recommendations': []
        }
        
        # Add cognitive score interpretation
        if mmse != 'N/A':
            if mmse >= 24:
                explanation['rationale'].append(
                    f"MMSE score of {mmse} indicates preserved cognitive function"
                )
            elif mmse >= 18:
                explanation['rationale'].append(
                    f"MMSE score of {mmse} suggests mild cognitive impairment"
                )
            else:
                explanation['rationale'].append(
                    f"MMSE score of {mmse} indicates significant cognitive impairment"
                )
        
        # Add MRI findings if available
        if 'attention_regions' in prediction_context:
            regions = prediction_context['attention_regions']
            explanation['rationale'].append(
                f"MRI analysis shows notable changes in: {', '.join(regions)}"
            )
        
        # Extract clinical guidelines from retrieved docs
        for doc in retrieved_docs[:3]:  # Top 3
            if doc['category'] in ['diagnostic_criteria', 'clinical_assessment']:
                explanation['clinical_guidelines'].append({
                    'source': doc['source'],
                    'excerpt': doc['text'][:200] + "..."
                })
        
        # Add recommendations based on diagnosis
        if 'AD' in label:
            explanation['recommendations'].extend([
                "Comprehensive neuropsychological evaluation",
                "Consider biomarker testing (CSF or PET)",
                "Discuss treatment options with neurologist",
                "Caregiver education and support",
                "Regular follow-up monitoring"
            ])
        elif 'MCI' in label:
            explanation['recommendations'].extend([
                "Annual cognitive monitoring",
                "Lifestyle interventions (exercise, diet, cognitive training)",
                "Address vascular risk factors",
                "Consider participation in clinical trials"
            ])
        else:  # CN
            explanation['recommendations'].extend([
                "Continue regular cognitive health monitoring",
                "Maintain healthy lifestyle",
                "Address modifiable risk factors"
            ])
        
        return explanation

# Initialize retriever
retriever = RAGRetriever(collection, embedding_model, top_k=5)

# Test retrieval
print("\nTesting RAG retrieval...")
test_query = {
    'predicted_label': 'AD',
    'confidence': 0.87,
    'mmse': 18,
    'moca': 16,
    'attention_regions': ['hippocampus', 'temporal_lobe']
}

retrieved = retriever.retrieve(test_query)
print(f"\n✓ Retrieved {len(retrieved)} relevant documents")
print("\nTop 3 Retrieved Documents:")
for i, doc in enumerate(retrieved[:3]):
    print(f"\n{i+1}. Source: {doc['source']}")
    print(f"   Category: {doc['category']}")
    print(f"   Text: {doc['text'][:150]}...")

# Generate explanation
explanation = retriever.generate_explanation(test_query, retrieved)
print("\n✓ Generated diagnostic explanation")

# ============================================================================
# SECTION 6: Integration with KnoAD-Net
# ============================================================================
print("\n[5/5] Integration Testing")
print("-" * 80)

def generate_rag_report(model_output, clinical_features, retriever):
    """
    Generate comprehensive RAG-augmented report
    
    Args:
        model_output: dict with model predictions
        clinical_features: dict with patient features
        retriever: RAGRetriever instance
    
    Returns:
        Complete diagnostic report
    """
    # Build prediction context
    pred_context = {
        'predicted_label': model_output['label'],
        'confidence': model_output['confidence'],
        'mmse': clinical_features.get('mmse', 'N/A'),
        'moca': clinical_features.get('moca', 'N/A'),
        'age': clinical_features.get('age', 'N/A'),
        'attention_regions': model_output.get('attention_regions', [])
    }
    
    # Retrieve relevant knowledge
    retrieved_docs = retriever.retrieve(pred_context, top_k=5)
    
    # Generate explanation
    explanation = retriever.generate_explanation(pred_context, retrieved_docs)
    
    # Format as comprehensive report
    report = f"""
{'='*80}
KNOADNET DIAGNOSTIC REPORT
{'='*80}

PATIENT INFORMATION
-------------------
Age:                {clinical_features.get('age', 'N/A')}
Education:          {clinical_features.get('education', 'N/A')} years
APOE4 Status:       {clinical_features.get('apoe4', 'N/A')} alleles

COGNITIVE ASSESSMENT
--------------------
MMSE Score:         {clinical_features.get('mmse', 'N/A')}/30
MoCA Score:         {clinical_features.get('moca', 'N/A')}/30
ADAS-Cog:           {clinical_features.get('adas_cog', 'N/A')}/70

MODEL PREDICTION
----------------
Diagnosis:          {explanation['diagnosis']}
Confidence:         {explanation['confidence']:.2%}

CLINICAL RATIONALE
------------------
"""
    for rationale in explanation['rationale']:
        report += f"• {rationale}\n"
    
    report += f"""
SUPPORTING CLINICAL GUIDELINES
-------------------------------
"""
    for i, guideline in enumerate(explanation['clinical_guidelines'][:2]):
        report += f"\n{i+1}. {guideline['source']}\n"
        report += f"   {guideline['excerpt']}\n"
    
    report += f"""
RECOMMENDATIONS
---------------
"""
    for rec in explanation['recommendations']:
        report += f"• {rec}\n"
    
    report += f"""
{'='*80}
This report was generated using KnoAD-Net with Retrieval-Augmented Generation.
For clinical use, please consult with a qualified healthcare professional.
{'='*80}
"""
    
    return report

# Test report generation
test_features = {
    'age': 72,
    'education': 16,
    'mmse': 18,
    'moca': 16,
    'adas_cog': 28,
    'apoe4': 1
}

test_prediction = {
    'label': 'Alzheimer\'s Disease (AD)',
    'confidence': 0.87,
    'attention_regions': ['hippocampus', 'temporal_lobe', 'entorhinal_cortex']
}

sample_report = generate_rag_report(test_prediction, test_features, retriever)
print("\nSample Diagnostic Report:")
print(sample_report)

# Save retriever and sample report
print("\nSaving RAG components...")

# Save retriever config
rag_config = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'collection_name': collection_name,
    'n_documents': len(chunks),
    'top_k': 5
}

with open(f"{config['directories']['rag_knowledge']}/rag_config.json", 'w') as f:
    json.dump(rag_config, f, indent=2)

# Save sample report
with open(f"{config['directories']['results']}/sample_rag_report.txt", 'w') as f:
    f.write(sample_report)

print(f"✓ Saved RAG configuration")
print(f"✓ Saved sample report")

print("\n" + "=" * 80)
print("RAG MODULE COMPLETE!")
print("=" * 80)
print(f"""
Summary:
--------
✓ Clinical knowledge base:  {len(clinical_documents)} documents
✓ Document chunks:           {len(chunks)} chunks
✓ Vector store:              ChromaDB with {len(chunks)} embeddings
✓ Retrieval system:          Ready for inference

Files Saved:
------------
✓ Vector store:      {config['directories']['rag_knowledge']}/chroma_db/
✓ RAG config:        {config['directories']['rag_knowledge']}/rag_config.json
✓ Sample report:     {config['directories']['results']}/sample_rag_report.txt

Next Step:
----------
Run Notebook 5: Comprehensive Evaluation & Paper Results
""")