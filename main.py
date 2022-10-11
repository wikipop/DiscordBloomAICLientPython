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
    konkretnie używam wersji bloom-1b7

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
    # Na etapie developmentu musiałem zmienić cache modelu, bo nie miałem miejsca na dysku C
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", use_cache=True, cache_dir="F:/.cache").to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", cache_dir="F:/.cache")
else:
    # Program jest przeznaczony do działania na moim serwerze (debian) a tam nie mam problemu z miejscem
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", use_cache=True).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(model.__class__.__name__)


def generate_prompt(user_input: str, result_len: int = 200) -> str:
    """
    Funkcja zwraca rezultat w postaci przedłużenia ciągu lub zdania.
    Używa modelu "BLOOM".

    :param user_input: Ciąg na podstawie którego generowany będzie rezultat
    :param result_len: Maksymalna długość rezultatu
    :return: Przedłużony ciąg lub zdanie
    """

    print(f"Zaczynam generowanie: {user_input} : {type(user_input)}")

    # Walidacja danych podanych przez użytkownika
    if not user_input:
        return "No user input specified"
        pass

    prompt: str = str(user_input)

    # Przygotowanie promptu
    input_ids = tokenizer(prompt, return_tensors="pt").to('cuda:0')
    # Generowanie rezultatu na podstawie podanego promptu
    sample = model.generate(**input_ids, max_length=result_len, top_k=1, temperature=0.9, repetition_penalty=2.0)
    # Dekodowanie i przypisanie rezultatu do zmiennej
    result: str = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    print(result)

    # Zwrot rezultatu
    return result


"""
    CLIENT
"""

# Deklaracja zamiarów aplikacji (boilerplate wymagany przez discord)
intents = discord.Intents.default()
intents.message_content = True


# Praktycznie cała ta klasa to boilerplate także nie widzę potrzeby rozpisywania się nad nią
class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.synced = False

    # Logika włączania bota i synchronizacji komend
    async def on_ready(self) -> None:
        await client.wait_until_ready()
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
async def generuj(interaction: discord.Interaction, prompt: str):
    # Deklaracja odpowiedzi (w przeciwnym wypadku skrypt ma 3 sekundy na odpowiedź)
    await interaction.response.defer()
    # Generowanie rezultatu
    result: str = generate_prompt(prompt)
    # Wysłanie rezultatu
    await interaction.followup.send(f"Rezultat: {result}")


# Inicjacja klienta (bota discord)
client.run(discord_token)
