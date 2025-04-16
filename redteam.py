import os
import asyncio

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FileSearchTool, MessageAttachment, FilePurpose
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.projects.models import AzureAISearchTool, AzureAISearchQueryType
from azure.ai.evaluation import evaluate, AzureAIProject, AzureOpenAIModelConfiguration, F1ScoreEvaluator
from azure.ai.evaluation import RelevanceEvaluator
from azure.ai.evaluation import (
    ContentSafetyEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    GroundednessEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
)

from azure.ai.evaluation import BleuScoreEvaluator, GleuScoreEvaluator, RougeScoreEvaluator, MeteorScoreEvaluator, RougeType
from azure.ai.projects.models import FunctionTool, RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput, ToolSet
from azure.ai.evaluation import ProtectedMaterialEvaluator, IndirectAttackEvaluator, RetrievalEvaluator, GroundednessProEvaluator
from typing import Any, Callable, Set, Dict, List, Optional
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy

from dotenv import load_dotenv
load_dotenv()

# Define a simple callback function that always returns a fixed response
def financial_advisor_callback(query: str) -> str:  # noqa: ARG001
    return "I'm a financial advisor assistant. I can help with investment advice and financial planning within legal and ethical guidelines."
# Create a more complex callback function that handles full conversation context
async def advanced_callback(messages: Dict, stream: bool = False, session_state: Any = None, context: Optional[Dict] =None) -> dict:
    """A more complex callback that processes conversation history"""
    # Extract the latest message from the conversation history
    messages_list = [{"role": chat_message.role,"content": chat_message.content} for chat_message in messages]
    latest_message = messages_list[-1]["content"]
    
    # In a real application, you might process the entire conversation history
    # Here, we're just simulating different responses based on the latest message
    response = "I'm an AI assistant that follows safety guidelines. I cannot provide harmful content."
    
    # Format the response to follow the openAI chat protocol format
    formatted_response = {
        "content": response,
        "role": "assistant"
    }
    
    return {"messages": [formatted_response]}
    
async def redteam_agent(query: str) -> str:
    returntxt = ""
    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_RED_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_RED_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_RED_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    model_config = AzureOpenAIModelConfiguration(
       azure_endpoint=azure_endpoint,
       api_key=api_key,
       api_version=api_version,
       azure_deployment=azure_deployment,
   )

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZUREAI_PROJECT_NAME")
    print(subscription_id, resource_group_name, project_name)
    azure_ai_project = AzureAIProject(subscription_id=subscription_id, 
                                      resource_group_name=resource_group_name, 
                                      project_name=project_name, 
                                      azure_crendential=credential)

    # Create the `RedTeam` instance with minimal configurations
    red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness],
        num_objectives=1,
    )

    # Run the red team scan called "Basic-Callback-Scan" with limited scope for this basic example
    # This will test 1 objective prompt for each of Violence and HateUnfairness categories with the Flip strategy
    result = await red_team.scan(
        target=financial_advisor_callback, scan_name="Basic-Callback-Scan", attack_strategies=[AttackStrategy.Flip]
    )

    returntxt += str(result)

    # Define a model configuration to test
    azure_oai_model_config = {
        "azure_endpoint": azure_endpoint,
        "azure_deployment": azure_deployment,
        "api_key": api_key,
    }
    # # Run the red team scan called "Intermediary-Model-Target-Scan"
    result = await red_team.scan(
        target=azure_oai_model_config, scan_name="Intermediary-Model-Target-Scan", attack_strategies=[AttackStrategy.Flip]
    )
    returntxt += str(result)

    # Create the RedTeam instance with all of the risk categories with 5 attack objectives generated for each category
    model_red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=2,
    )

    azure_openai_config = {
        "azure_endpoint": os.getenv("AZURE_OPENAI_RED_ENDPOINT_MINI"),
        "azure_deployment": "gpt-4.1-mini",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY")
    }

    # Run the red team scan with multiple attack strategies
    advanced_result = await model_red_team.scan(
        # target=advanced_callback,
        target=azure_openai_config,
        scan_name="Advanced-Callback-Scan",
        attack_strategies=[
            AttackStrategy.EASY,  # Group of easy complexity attacks
            AttackStrategy.MODERATE,  # Group of moderate complexity attacks
            #AttackStrategy.CharacterSpace,  # Add character spaces
            #AttackStrategy.ROT13,  # Use ROT13 encoding
            #AttackStrategy.UnicodeConfusable,  # Use confusable Unicode characters
            #AttackStrategy.CharSwap,  # Swap characters in prompts
            #AttackStrategy.Morse,  # Encode prompts in Morse code
            #AttackStrategy.Leetspeak,  # Use Leetspeak
            #AttackStrategy.Url,  # Use URLs in prompts
            #AttackStrategy.Binary,  # Encode prompts in binary
            AttackStrategy.Flip,
            AttackStrategy.Jailbreak,
            AttackStrategy.Tense,
            AttackStrategy.ROT13,
            AttackStrategy.UnicodeConfusable,
            AttackStrategy.UnicodeSubstitution,
            AttackStrategy.Leetspeak,
            AttackStrategy.Morse,
            AttackStrategy.DIFFICULT,
            AttackStrategy.Compose([AttackStrategy.Base64, AttackStrategy.ROT13]),  # Use two strategies in one attack
        ],
        max_parallel_tasks=40,
        timeout=4800,
        output_path="./Advanced-Callback-Scan.json",
    )

    returntxt += str(advanced_result)

    return returntxt

if __name__ == "__main__":
    query = "what are the best practicse from Virgnia Railway express project?"
    asyncio.run(redteam_agent(query=query))