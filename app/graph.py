from google import genai
client = genai.Client(api_key="AIzaSyDsUxVsxS4Dg17K55T5qW9H-8y03bgeZO4")


SYSTEM_PROMPT="""

"""

user = input("Hey — tell me, how can I help?  ")

res = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[
        {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "user",   "parts": [{"text": user}]}
    ]
)


print("🤖", res.text)
