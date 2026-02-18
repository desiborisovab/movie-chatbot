from src.inference import MovieRAGBot

RUN_ID = ""

bot = MovieRAGBot(RUN_ID)

question = "Tell me about Kansas Saloon Smashers."
response = bot.answer(spark, question)

print(response)
