from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dotenv import load_dotenv
from discord import app_commands
import discord
import torch
import os
load_dotenv()
discord_token: str = os.environ.get("TOKEN")
if os.name == 'nt':
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", use_cache=True, cache_dir="F:/.cache")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", cache_dir="F:/.cache")
else:
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(model.__class__.__name__)
def generate_prompt(user_input: str, result_len: int = 200) -> str:
    print(f"Zaczynam generowanie: {user_input} : {type(user_input)}")
    if not user_input:
        return "No user input specified"
        pass
    prompt: str = str(user_input)
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=200, top_k=1, temperature=0.9, repetition_penalty=2.0)
    result: str = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    print(result)
    return result
intents = discord.Intents.default()
intents.message_content = True
class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.synced = False
    async def on_ready(self) -> None:
        await client.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print(f"Logged in as {self.user}.")
client = MyClient()
tree = app_commands.CommandTree(client)
@tree.command(name="generuj", description="generowanie rezultatu przy użyciu modelu Bloom")
async def generuj(interaction: discord.Interaction, prompt: str):
    await interaction.response.send_message(f"Rezultat: {generate_prompt(prompt)}")
client.run(discord_token)