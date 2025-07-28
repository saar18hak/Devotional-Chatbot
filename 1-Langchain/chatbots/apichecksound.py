import requests
import streamlit as st

st.title("🛕 Radha Rani Devotional Chatbot with Voice")

# Ask the user for a question
user_prompt = st.text_input("🙏 Ask Radha Rani's guidance:")

if user_prompt:
    if st.button("✨ Get Divine Response"):
        # ElevenLabs API settings
        api_key = "sk_04d47fc7e33d48f0a041b102e4d3af645089115159fd69ab"  # Replace with your actual key
        voice_id = "ttpam6l3Fgkia7uX33b6"    # Replace with actual voice ID

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

        data = {
            "text": user_prompt,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.75}
        }

        # Request audio generation
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Save the audio
            with open("output.mp3", "wb") as f:
                f.write(response.content)
            st.success("✅ Voice generated!")
            st.markdown(f"**Radha Rani says:** {user_prompt}")
            st.audio("output.mp3", format="audio/mp3")
        else:
            st.error("❌ Failed to generate voice. Please check API key and voice ID.")
