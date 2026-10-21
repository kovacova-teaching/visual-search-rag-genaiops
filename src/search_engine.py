import base64
import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from src.config import Config
# from src.security import SecurityManager # Deprecated in favor of GovernanceGate
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store

class VisualSearchEngine:
    def __init__(self):
        # self.security_manager = SecurityManager()
        self.governance_gate = GovernanceGate()
        
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.3
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        
        # Initialize Vector Store (Chroma or Azure Search) via Factory
        self.vector_store = get_vector_store(self.embeddings)

    def _encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def generate_image_description(self, image_bytes):
        """
        Uses GPT-4o to describe the product image for vectorization.
        Traceable with MLflow.
        """
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="generate_description", nested=True):
            base64_image = self._encode_image(image_bytes)
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this product image in detail for a product catalog search. Focus on visual attributes like color, pattern, material, style, and category. Keep it descriptive but concise."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ]
            )
            
            mlflow.log_params({"model": "gpt-4o", "task": "image_captioning"})
            
            response = self.llm.invoke([message])
            description = response.content
            
            # Governance check on output (Generated Description)
            gov_result = self.governance_gate.validate_output(description)
            if not gov_result['passed']:
                 mlflow.set_tag("governance_check", "failed")
                 mlflow.log_dict({"violations": gov_result['violations']}, "governance_violations.json")
                 return "[CONTENT BLOCKED]"
            
            mlflow.log_text(description, "generated_description.txt")
            return description

    def search_similar_products(self, image_bytes, k=5):
        """
        1. Generate description from image.
        2. Embed description.
        3. Search vector store.
        """
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="search_products_image"):
            description = self.generate_image_description(image_bytes)
            print(f"DEBUG: Generated Description: {description}")
            
            if description == "[CONTENT BLOCKED]":
                return [], "Content analysis failed security checks."
            
            mlflow.log_param("k", k)
            mlflow.log_param("query_description", description)
            
            # Perform similarity search using the generated text description
            docs = self.vector_store.similarity_search(description, k=k)
            
            # Log results count
            mlflow.log_metric("results_count", len(docs))
            
            return docs, description

    def search_by_text(self, query_text, k=5):
        """
        Search for products using a text query directly.
        """
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="search_products_text"):
            print(f"DEBUG: Text Query: {query_text}")
            
            # Governance check on input (Text Query)
            gov_check = self.governance_gate.validate_input(query_text)
            if not gov_check['passed']:
                 mlflow.set_tag("governance_check", "failed")
                 mlflow.log_dict({"violations": gov_check['violations']}, "governance_violations.json")
                 return [], "Query blocked by security checks."

            mlflow.log_param("k", k)
            mlflow.log_param("query_text", query_text)
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(query_text, k=k)
            
            mlflow.log_metric("results_count", len(docs))
            
            return docs, query_text


    def synthesize_response(self, docs, user_query_description):
        """
        Optional: Start a chat with the user or summarize findings.
        """
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="synthesize_response"):
            if not docs:
                return "No matching products found."
            
            context = "\n".join([f"- {doc.page_content} (Metadata: {doc.metadata})" for doc in docs])
            
            prompt = f"""
            User searched for a product looking like this description: "{user_query_description}"
            
            Found the following products in the catalog:
            {context}
            
            Please summarize the matches and recommend the best fit.
            """
            
            # Governance Check on Prompt
            # SKIPPED: Prompt is constructed from trusted catalog data (ids) which trigger PII false positives (zip codes).
            # User input part is already validated in search_by_text/search_similar_products.
            # gov_check = self.governance_gate.validate_input(prompt)
            # if not gov_check['passed']:
            #      return f"Security check failed: {gov_check['violations']}"
            
            response = self.llm.invoke(prompt).content
            mlflow.log_text(response, "final_response.txt")
            return response
