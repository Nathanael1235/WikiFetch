from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a professional dancer."},
    {"role": "user", "content": "explain why dancing is so competitive."}
  ]
)

print(completion.choices[0].message)

