from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dotenv import load_dotenv
from discord import app_commands
import discord
import torch
import os

"""
    Program wykorzystuje bota na popularnej platformie discord jako klienta do modelu sztucznej inteligencji "bloom".

    Bloom, BigScience Large Open-science Open-access Multilingual Language Model
    model: https://bigscience.huggingface.co/blog/bloom

    autor: wikipop (https://github.com/wikipop)
    
    Skrypt wymaga pliku .env z zapisanym tokenem bota discord w postaci
    TOKEN=xxx
"""

"""
    SERVER
"""

# Wczytywanie zmiennych środowiskowych z pliku .env
load_dotenv()
discord_token: str = os.environ.get("TOKEN")

# Inicjacja modelu "BLOOM"
if os.name == 'nt':
    # Na etapie developmentu musiałem zmienić cache modelu bo nie miałem miejsca na dysku C
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", use_cache=True, cache_dir="F:/.cache")
    tokenizer = AutoTokenizer.from_pretrained("bbigscience/bloom", cache_dir="F:/.cache")
else:
    # Program jest przeznaczony do działania na moim serwerze (debian) a tam nie mam problemu z miejscem
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained("bbigscience/bloom")

torch.set_default_tensor_type(torch.FloatTensor)
print(model.__class__.__name__)


def generate_prompt(user_input: str, result_len: int = 200) -> str:
    """
    Funkcja ma za zadanie przygotowanie i zwrócenie rezultatu na podstawie
    dwóch podanych przez użytkownika parametrów. Używa modelu "BLOOM".

    :param user_input: str
        Prompt na którego podstawie generowany będzie rezultat
    :param result_len: int = 200
        Maksymalna długość rezultatu
    :return: str
    """

    # Walidacja danych podanych przez użytkownika
    if not user_input:
        return "No user input specified"
        pass

    prompt: str = user_input

    # Przygotowanie promptu
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    # Generowanie rezultatu na podstawie podanego promptu
    sample = model.generate(**input_ids, max_length={result_len}, top_k=1, temperature=0.9, repetition_penalty=2.0)
    # Dekodowanie i przypisanie rezultatu do zmiennej
    result: str = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    print(result)

    # Zwrot rezultatu
    return result


"""
    CLIENT
"""
# Tworzenie zamiarów bota (boilerplate wymagany przez discord)
intents = discord.Intents.default()
intents.message_content = True


# Praktycznie cała ta klasa to boilerplate także nie widzę potrzeby rozpisywania się nad nią
class MyClient(discord.Client):
    def __init__(self):
        # Przekazywanie zamiarów
        super().__init__(intents=intents)
        self.synced = False

    # Logika włączania bota i synchronizacji komend
    async def on_ready(self) -> None:
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print(f"Logged in as {self.user}.")


# Tworzenie instancji Klienta (klasy MyClient)
client = MyClient()
# Tworzenie "drzewa komend" czyli boilerplate
tree = app_commands.CommandTree(client)


# Logika pobierania promptu użytkownika i wyświetlania go.
@tree.command(name="generuj", description="generowanie rezultatu przy użyciu modelu Bloom")
async def generuj(interaction: discord.Interaction, prompt: str, result_length: int):
    await interaction.response.send_message(f"Rezultat: {generate_prompt(prompt, result_length)}")


# Inicjacja klienta (bota discord)
intents = discord.Intents.default()
intents.message_content = True
client = MyClient()
client.run(discord_token)
