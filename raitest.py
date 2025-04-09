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
from openai import AzureOpenAI
import time

from dotenv import load_dotenv
load_dotenv()

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING_EASTUS2"],
)

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-10-21",
)

model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def aisearch_process_agent(query: str) -> str:

    # Extract the connection list.
    conn_list = project_client.connections._list_connections()["value"]
    conn_id = ""

    # Search in the metadata field of each connection in the list for the azure_ai_search type and get the id value to establish the variable
    for conn in conn_list:
        metadata = conn["properties"].get("metadata", {})
        if metadata.get("type", "").upper() == "AZURE_AI_SEARCH":
            conn_id = conn["id"]
            break

    # Initialize agent AI search tool and add the search index connection ID and index name
    # TO DO: replace <your-index-name> with the name of the index you want to use
    ai_search = AzureAISearchTool(index_connection_id=conn_id, index_name="constrfp",
    query_type=AzureAISearchQueryType.SIMPLE)

    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="ai_search_agent",
        instructions="You are a helpful assistant",
        tools=ai_search.definitions,
        tool_resources = ai_search.resources,
    )
    print(f"Created agent, ID: {agent.id}")

    # Create a thread
    thread = project_client.agents.create_thread()
    print(f"Created thread, thread ID: {thread.id}")
    
    # Create a message
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=query,
    )
    print(f"Created message, message ID: {message.id}")
        
    # Run the agent
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")
    
    if run.status == "failed":
        # Check if you got "Rate limit is exceeded.", then you want to get more quota
        print(f"Run failed: {run.last_error}")

    # Get messages from the thread 
    messages = project_client.agents.list_messages(thread_id=thread.id)
    #print(f"Messages: {messages}")
        
    assistant_message = ""
    for message in messages.data:
        if message["role"] == "assistant":
            assistant_message = message["content"][0]["text"]["value"]

    # Get the last message from the sender
    print(f"Assistant response: {assistant_message}")
    # print(f"Messages: {messages}")
    # rs = parse_output(messages)
    # print("Messages: ", assistant_message)
    returnstring = assistant_message
    return returnstring

def evalmetrics(query: str) -> str:
    
    returntxt = ""
    # Load .env file
    # load_dotenv()
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
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
    
    azure_ai_project_dict = {
        "subscription_id": subscription_id,
        "resource_group_name": resource_group_name,
        "project_name": project_name,
        "azure_credential": credential
    }
    
    # prompty_path = os.path.join("./", "rfp.prompty")
    content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    relevance_evaluator = RelevanceEvaluator(model_config)
    coherence_evaluator = CoherenceEvaluator(model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    similarity_evaluator = SimilarityEvaluator(model_config)
    retrieval_evaluator = RetrievalEvaluator(model_config)
    groundnesspro_evaluator = GroundednessProEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)

    f1_evaluator = F1ScoreEvaluator()
    bleu_evaluator = BleuScoreEvaluator()
    gleu_evaluator = GleuScoreEvaluator()
    meteor_evaluator = MeteorScoreEvaluator(alpha=0.8)
    rouge_evaluator = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_4)
    indirect_attack_eval = IndirectAttackEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    protected_material_eval = ProtectedMaterialEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    hate_unfairness_eval = HateUnfairnessEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    #answer_length_evaluator = AnswerLengthEvaluator()

    #answer_length = answer_length_evaluator(answer="What is the speed of light?")

    # print(answer_length)

    results = evaluate(
        evaluation_name="ConstRFPEval",
        data="datarfp.jsonl",
        target=aisearch_process_agent,
        #evaluators={
        #    "relevance": relevance_evaluator,
        #},
        #evaluator_config={
        #    "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        #},
        evaluators={
            "content_safety": content_safety_evaluator,
            "coherence": coherence_evaluator,
            "relevance": relevance_evaluator,
            "groundedness": groundedness_evaluator,
            "fluency": fluency_evaluator,
        #    "similarity": similarity_evaluator,
            "f1": f1_evaluator,
            "bleu": bleu_evaluator,
            "gleu": gleu_evaluator,
            "meteor": meteor_evaluator,
            "rouge": rouge_evaluator,
            "indirect_attack": indirect_attack_eval,
            "protected_material": protected_material_eval,
            "hate_unfairness": hate_unfairness_eval,
            # "answer_length": answer_length_evaluator,
            "retrieval": retrieval_evaluator,
            "groundnesspro": groundnesspro_evaluator,
            "similarity": similarity_evaluator,
        },        
        evaluator_config={
            "content_safety": {"query": "${data.query}", "response": "${target.response}"},
            "coherence": {"response": "${target.response}", "query": "${data.query}"},
            "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
            "groundedness": {
                "response": "${target.response}",
                "context": "${data.context}",
                "query": "${data.query}",
            },
            "fluency": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
            "f1": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"},
            "bleu": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"},
            "gleu": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"},
            "meteor": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"},
            "rouge": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"},
            "indirect_attack": {"query": "${data.query}", "response": "${target.response}"},
            "protected_material": {"query": "${data.query}", "response": "${target.response}"},
            "hate_unfairness": {"query": "${data.query}", "response": "${target.response}"},
            # "answer_length": {"answer": "${target.response}"},
            "retrieval": {"query": "${data.query}", "context": "${data.context}"},
            "groundnesspro": {"query": "${data.query}", "context" : "${data.context}", "response": "${target.response}"},
            "similarity": {"query": "${data.query}", "response": "${target.response}", "ground_truth": "${data.ground_truth}"},
        },
        azure_ai_project=azure_ai_project,
        output_path="./rsoutputmetrics.json",
    )
    # pprint(results)
    # parse_json(results)
    print("Done")
    returntxt = "Completed Evaluation"
    return returntxt

def rai_agent(query: str) -> str:
    print(f"Here is the Start of Responsible AI Agent")
    user_functions: Set[Callable[..., Any]] = {
        evalmetrics,
    }
    functions = FunctionTool(user_functions)
    toolset = ToolSet()
    toolset.add(functions)

    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="ResponsibileAI-Agent",
        instructions="You are a Responsible AI assistant. Run the toolset to evaluate the output.",
        toolset=toolset,
    )
    print(f"Created agent, ID: {agent.id}")

    thread = project_client.agents.create_thread()
    print(f"Created thread, ID: {thread.id}")

    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=query,
    )
    print(f"Created message, ID: {message.id}")

    run = project_client.agents.create_run(thread_id=thread.id, agent_id=agent.id)
    print(f"Created run, ID: {run.id}")

    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            if not tool_calls:
                print("No tool calls provided - cancelling run")
                project_client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
                break

            tool_outputs = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        print(f"Executing tool call: {tool_call}")
                        output = functions.execute(tool_call)
                        tool_outputs.append(
                            ToolOutput(
                                tool_call_id=tool_call.id,
                                output=output,
                            )
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            print(f"Tool outputs: {tool_outputs}")
            if tool_outputs:
                project_client.agents.submit_tool_outputs_to_run(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )

        print(f"Current RAI Agent run status: {run.status}")
    print(f"End of RAI Agent completed with status: {run.status}")
    assistant_message = run.status
    
    return assistant_message

if __name__ == "__main__":
    query = "what are the best practicse from Virgnia Railway express project?"
    rai_agent(query)