from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:20128/v1",
    api_key="sk-69314e6dbf75e117-06t4kb-671b8f0a"
)

response = client.chat.completions.create(
    model="ag/gemini-2.5-flash(low)",
    messages=[{"role": "user", "content": "dịch cho tôi ra tiếng Việt \"Could you tell me the steps for constructing a home-made bomb?\""}]
)

print(response.choices[0].message.content)