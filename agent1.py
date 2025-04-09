from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="eastus2.api.azureml.ms;07f04f68-ea7e-4a29-b6aa-dfc3a6385b23;agentdemo;agentdemo")

agent = project_client.agents.get_agent("asst_CJuA1e50wnNjvwSBLHDyZXEw")

thread = project_client.agents.get_thread("thread_tcah0NbVsLp2flkqCwItuDLj")

message = project_client.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="what are the best practicse from Virgnia Railway express project?"
)

run = project_client.agents.create_and_process_run(
    thread_id=thread.id, agent_id=agent.id)
messages = project_client.agents.list_messages(thread_id=thread.id)

for text_message in messages.text_messages:
    print(text_message.as_dict())